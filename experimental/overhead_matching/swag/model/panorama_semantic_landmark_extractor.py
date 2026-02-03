"""Extractor for panorama-based semantic landmarks.

This module contains PanoramaSemanticLandmarkExtractor which extracts embeddings
for landmark descriptions from panorama images.
"""

import common.torch.load_torch_deps
import torch
import pickle
import json
from pathlib import Path
from experimental.overhead_matching.swag.model.swag_config_types import (
    PanoramaSemanticLandmarkExtractorConfig,
    ExtractorDataRequirement)
from experimental.overhead_matching.swag.model.swag_model_input_output import (
    ModelInput, ExtractorOutput)
from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    load_all_jsonl_from_folder, make_embedding_dict_from_json, make_sentence_dict_from_pano_jsons)
from experimental.overhead_matching.swag.model.additional_panorama_extractors import (
    yaw_angles_to_binary_vector,
    extract_yaw_angles_from_bboxes,
    load_v2_pickle,
    normalize_cropped_embeddings,
)


def _extract_yaw_angles_from_bboxes(bounding_boxes: list[dict]) -> list[float]:
    """Extract valid yaw angles from a list of bounding boxes.

    Args:
        bounding_boxes: List of bounding box dicts, each with a "yaw_angle" key.

    Returns:
        List of valid yaw angles (0, 90, 180, or 270) as floats.
    """
    valid_yaws = {'0', '90', '180', '270'}
    yaw_angles = []
    for bbox in bounding_boxes:
        yaw_str = bbox.get("yaw_angle", "")
        if yaw_str in valid_yaws:
            yaw_angles.append(float(yaw_str))
    return yaw_angles


def _load_v2_pickle(pickle_path: Path) -> dict | None:
    """Load a v2.0 format pickle file.

    Args:
        pickle_path: Path to the embeddings.pkl file.

    Returns:
        The loaded dict if it's v2.0 format, None otherwise.
    """
    if not pickle_path.exists():
        return None
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and not data.get("version") == "2.0":
        raise RuntimeError("Embedding pickle was not version 2")
    return data


def _iter_city_directories(base_path: Path):
    """Iterate over city directories in a base path.

    Args:
        base_path: Path containing city subdirectories.

    Yields:
        Tuple of (city_name, city_dir) for each city directory found.

    Raises:
        FileNotFoundError: If base_path doesn't exist or contains no city directories.
    """
    if not base_path.exists():
        raise FileNotFoundError(f"Embedding base path does not exist: {base_path}")

    city_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not city_dirs:
        raise FileNotFoundError(f"No city directories found in {base_path}")

    for city_dir in city_dirs:
        yield city_dir.name, city_dir


def _normalize_cropped_embeddings(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Re-normalize embeddings after cropping to a shorter length.

    Args:
        features: Tensor of shape (batch, num_items, embedding_dim).
        mask: Boolean tensor of shape (batch, num_items), True = masked/invalid.

    Returns:
        Features tensor with valid embeddings normalized.
    """
    valid_mask = ~mask
    if valid_mask.any():
        valid_features = features[valid_mask]
        features[valid_mask] = valid_features / torch.norm(valid_features, dim=-1, keepdim=True)
    return features


def load_embedding_type_across_cities(
    base_path: Path,
    embedding_key: str,
    id_to_idx_key: str,
    default_dim: int,
    dedupe_keys: bool = False,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Load specific embedding type from all city v2.0 pickles with offset handling.

    Args:
        base_path: Path containing city subdirectories.
        embedding_key: Key for the embedding tensor in pickle (e.g., "proper_noun_embeddings").
        id_to_idx_key: Key for the id->index dict in pickle (e.g., "proper_noun_to_idx").
        default_dim: Default embedding dimension if no embeddings found.
        dedupe_keys: If True, skip duplicate keys across cities (for shared vocab like location types).

    Returns:
        Tuple of (concatenated embedding tensor, combined id_to_idx dict with offsets applied).
        Returns (empty tensor, empty dict) if no cities have the requested embedding type.
    """
    all_tensors = []
    all_id_to_idx = {}

    for city_name, city_dir in _iter_city_directories(base_path):
        pickle_path = city_dir / "embeddings" / "embeddings.pkl"
        data = _load_v2_pickle(pickle_path)
        if data is None:
            print(f"  Warning: {pickle_path} missing or not v2.0 format, skipping")
            continue

        if embedding_key not in data or id_to_idx_key not in data:
            print(f"  Warning: No {embedding_key} in {pickle_path}, skipping")
            continue

        tensor = data[embedding_key]
        id_to_idx = data[id_to_idx_key]

        offset = len(all_id_to_idx)
        for k, v in id_to_idx.items():
            if dedupe_keys and k in all_id_to_idx:
                continue
            all_id_to_idx[k] = v + offset

        all_tensors.append(tensor)
        print(f"  Loaded {len(id_to_idx)} {embedding_key} for {city_name}")

    if all_tensors:
        return torch.cat(all_tensors, dim=0), all_id_to_idx
    return torch.zeros((0, default_dim)), {}


def extract_panorama_data_across_cities(
    base_path: Path,
    extract_fn,  # Callable[[str, dict], T | None]
) -> dict:
    """Extract per-panorama data from all city pickles using custom extractor function.

    Args:
        base_path: Path containing city subdirectories.
        extract_fn: Function (pano_id_clean, pano_data) -> value or None.
                    Return None to skip this panorama.

    Returns:
        Dict mapping pano_id -> extracted value.
    """
    result = {}

    for city_name, city_dir in _iter_city_directories(base_path):
        pickle_path = city_dir / "embeddings" / "embeddings.pkl"
        data = _load_v2_pickle(pickle_path)
        if data is None:
            print(f"  Warning: {pickle_path} missing or not v2.0 format, skipping")
            continue

        if "panoramas" not in data:
            print(f"  Warning: No panoramas in {pickle_path}, skipping")
            continue

        for pano_id, pano_data in data["panoramas"].items():
            pano_id_clean = pano_id.split(",")[0]
            value = extract_fn(pano_id_clean, pano_data)
            if value is not None:
                result[pano_id_clean] = value

    return result


class PanoramaSemanticLandmarkExtractor(torch.nn.Module):
    """
    Extractor for panorama-based semantic landmarks.

    Unlike SemanticLandmarkExtractor which matches landmarks by OSM properties,
    this extractor works with landmarks extracted directly from panorama images
    by vision models. Landmarks are associated with panoramas by ID alone.

    Supports both v1.0 format (separate files for embeddings, sentences, metadata)
    and v2.0 format (single pickle with all data).
    """

    def __init__(self, config: PanoramaSemanticLandmarkExtractorConfig, semantic_embedding_base_path: Path):
        super().__init__()
        self.config = config
        self.semantic_embedding_base_path = Path(semantic_embedding_base_path).expanduser()
        self.files_loaded = False
        self.all_embeddings = None  # dict: custom_id -> embedding (list or tensor)
        self.all_sentences = None  # dict: custom_id -> sentence string
        self.panorama_metadata = None  # dict: pano_id -> list of landmark metadata dicts
        self._found_pano_ids = set()
        self._missing_pano_ids = set()

    def load_files(self):
        """Load embeddings, sentences, and metadata from multi-city directory structure."""
        base_path = self.semantic_embedding_base_path / self.config.embedding_version

        if not base_path.exists():
            raise FileNotFoundError(f"Embedding base path does not exist: {base_path}")

        # Find all city directories
        city_dirs = [d for d in base_path.iterdir() if d.is_dir()]
        if not city_dirs:
            raise FileNotFoundError(f"No city directories found in {base_path}")

        self.all_embeddings = {}
        self.all_sentences = {}
        self.panorama_metadata = {}

        for city_dir in city_dirs:
            city_name = city_dir.name
            print(f"Loading panorama landmarks for city: {city_name}")

            # Check for v2.0 pickle format first
            pickle_path = city_dir / "embeddings" / "embeddings.pkl"
            data = load_v2_pickle(pickle_path)

            if data is not None:
                # v2.0 format
                city_embeddings, city_sentences, new_metadata = self._load_v2_data(data)

                # Assert no key overlap - custom_ids include pano_id so should be unique across cities
                embedding_overlap = set(self.all_embeddings.keys()) & set(city_embeddings.keys())
                assert not embedding_overlap, f"Unexpected embedding key overlap: {embedding_overlap}"
                sentence_overlap = set(self.all_sentences.keys()) & set(city_sentences.keys())
                assert not sentence_overlap, f"Unexpected sentence key overlap: {sentence_overlap}"
                metadata_overlap = set(self.panorama_metadata.keys()) & set(new_metadata.keys())
                assert not metadata_overlap, f"Unexpected metadata key overlap: {metadata_overlap}"

                self.all_embeddings.update(city_embeddings)
                self.all_sentences.update(city_sentences)
                self.panorama_metadata.update(new_metadata)
                print(f"  Loaded {len(city_embeddings)} embeddings from v2.0 pickle")
                print(f"  Loaded {len(city_sentences)} sentences from v2.0 pickle")
                print(f"  Loaded metadata for {len(new_metadata)} panoramas from v2.0 pickle")
                continue
            else:  # v1 format
                # v1.0 format: try to load from pickle or JSONL
                if pickle_path.exists():
                    with open(pickle_path, 'rb') as f:
                        embedding_data = pickle.load(f)
                    # v1.0 pickle format: (tensor, id_to_idx) tuple
                    tensor, id_to_idx = embedding_data
                    city_embeddings = {}
                    for custom_id, idx in id_to_idx.items():
                        city_embeddings[custom_id] = tensor[idx].tolist()
                    self.all_embeddings.update(city_embeddings)
                    print(f"  Loaded {len(city_embeddings)} embeddings from v1.0 pickle")
                else:
                    # v1.0 JSONL fallback
                    embedding_dir = city_dir / "embeddings"
                    if embedding_dir.exists():
                        city_embeddings = make_embedding_dict_from_json(
                            load_all_jsonl_from_folder(embedding_dir))
                        self.all_embeddings.update(city_embeddings)
                        print(f"  Loaded {len(city_embeddings)} embeddings from JSONL")


                # v1.0 format: Load sentences from separate directory
                sentence_dir = city_dir / "sentences"
                metadata_from_sentences = None
                if sentence_dir.exists():
                    city_sentences, metadata_from_sentences, _ = make_sentence_dict_from_pano_jsons(
                        load_all_jsonl_from_folder(sentence_dir))
                    self.all_sentences.update(city_sentences)
                    print(f"  Loaded {len(city_sentences)} sentences")

                # v1.0 format: Load panorama metadata from separate file
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

        assert len(self.all_embeddings) > 0, f"Failed to load any embeddings from {base_path}"
        first_embedding = next(iter(self.all_embeddings.values()))
        embedding_len = len(first_embedding) if isinstance(first_embedding, list) else first_embedding.shape[0]
        assert embedding_len >= self.config.openai_embedding_size, \
            f"Requested embedding length ({self.config.openai_embedding_size}) longer than available ({embedding_len})"

        # Remove coordinates from keys in self.panorama_metadata
        self.panorama_metadata = {k.split(",")[0]: v for k, v in self.panorama_metadata.items()}

        self.files_loaded = True
        print(f"Total embeddings loaded: {len(self.embedding_id_to_idx)}")
        print(f"Total panoramas with landmarks: {len(self.panorama_metadata)}")

    def _load_v2_data(self, data: dict) -> tuple[dict, dict, dict]:
        """Load embeddings, sentences, and metadata from v2.0 pickle format.

        Args:
            data: The loaded v2.0 pickle dict.

        Returns:
            Tuple of (embeddings_dict, sentences_dict, metadata_dict).
        """
        # Extract embeddings using dict-based approach (avoids offset bugs)
        tensor = data["description_embeddings"]
        id_to_idx = data["description_id_to_idx"]
        embeddings = {}
        for custom_id, idx in id_to_idx.items():
            embeddings[custom_id] = tensor[idx]

        # Extract sentences and metadata from panoramas
        sentences = {}
        metadata = {}
        for pano_id, pano_data in data.get("panoramas", {}).items():
            pano_id_clean = pano_id.split(",")[0]
            if pano_id_clean not in metadata:
                metadata[pano_id_clean] = []

            for landmark in pano_data.get("landmarks", []):
                landmark_idx = landmark["landmark_idx"]
                description = landmark["description"]
                custom_id = f"{pano_id}__landmark_{landmark_idx}"

                # Store sentence
                sentences[custom_id] = description

                # Extract yaw angles from bounding boxes using shared function
                yaw_angles = extract_yaw_angles_from_bboxes(
                    landmark.get("bounding_boxes", []))

                metadata[pano_id_clean].append({
                    "landmark_idx": landmark_idx,
                    "custom_id": custom_id,
                    "yaw_angles": yaw_angles,
                })

        return embeddings, sentences, metadata

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
                continue
            matching_landmarks = self.panorama_metadata[pano_id]
            num_landmarks = len(matching_landmarks)
            valid_landmarks.append(num_landmarks)

        max_num_landmarks = max(valid_landmarks) if valid_landmarks else 0

        # Initialize output tensors
        mask = torch.ones((batch_size, max_num_landmarks), dtype=torch.bool)
        features = torch.zeros((batch_size, max_num_landmarks, self.config.openai_embedding_size))
        positions = torch.zeros((batch_size, max_num_landmarks, 2, 2))

        max_description_length = 0

        # Process each batch item
        for i, item in enumerate(model_input.metadata):
            pano_id = item['pano_id']

            # Find matching panorama metadata
            if pano_id not in self.panorama_metadata:
                self._missing_pano_ids.add(pano_id)
                continue
            self._found_pano_ids.add(pano_id)
            matching_landmarks = self.panorama_metadata[pano_id]

            # Sort by landmark_idx to ensure consistent ordering
            matching_landmarks = sorted(matching_landmarks, key=lambda x: x["landmark_idx"])

            for landmark_idx, landmark_meta in enumerate(matching_landmarks):
                custom_id = landmark_meta["custom_id"]
                yaw_angles = landmark_meta["yaw_angles"]

                # Get embedding
                if custom_id not in self.embedding_id_to_idx:
                    print(f"Warning: missing embedding for {custom_id}")
                    continue

                # Crop embedding if needed
                embedding = self.all_embeddings[custom_id]
                if isinstance(embedding, list):
                    embedding = torch.tensor(embedding)
                features[i, landmark_idx, :] = embedding[:self.config.openai_embedding_size]

                # Convert yaw angles to binary presence vector
                yaw_vector = yaw_angles_to_binary_vector(yaw_angles)

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

        # Re-normalize embeddings if we cropped them
        features = normalize_cropped_embeddings(features, mask)

        debug = {}

        # Create debug tensor for sentences
        sentence_debug = torch.zeros(
            (batch_size, max_num_landmarks, max(max_description_length, 1)), dtype=torch.uint8)

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
                sentence_debug[i, landmark_idx, :len(sentence_bytes)] = sentence_tensor
        debug["sentences"] = sentence_debug.to(model_input.image.device)

        return ExtractorOutput(
            features=features.to(model_input.image.device),
            mask=mask.to(model_input.image.device),
            positions=positions.to(model_input.image.device),
            debug=debug)

    @property
    def output_dim(self):
        return self.config.openai_embedding_size

    @property
    def num_position_outputs(self):
        return 2

    @property
    def data_requirements(self) -> list[ExtractorDataRequirement]:
        # This extractor doesn't use landmarks from the dataset
        # (it uses vision-extracted landmarks stored separately)
        return []

    def get_pano_id_stats(self) -> tuple[int, int]:
        """Return (num_found, num_missing) pano IDs encountered during forward passes."""
        return len(self._found_pano_ids), len(self._missing_pano_ids)

    def print_pano_id_summary(self):
        """Print a summary of found vs missing pano IDs. Call after processing is complete."""
        num_found = len(self._found_pano_ids)
        num_missing = len(self._missing_pano_ids)
        total = num_found + num_missing
        if num_missing > 0:
            print(f"\n{'!'*80}")
            print(f"WARNING: {num_missing}/{total} pano IDs ({100*num_missing/total:.1f}%) not found in embedding metadata!")
            print(f"  Found: {num_found}, Missing: {num_missing}")
            print(f"{'!'*80}\n")
        else:
            print(f"All {num_found} pano IDs found in embedding metadata.")


class PanoramaProperNounExtractor(torch.nn.Module):
    """
    Extractor for proper noun embeddings from panorama landmarks.

    Proper nouns are business names, street signs, etc. extracted from panorama images.
    Produces one token per proper noun per landmark. Many landmarks may have zero proper nouns.
    """

    def __init__(self, config: PanoramaProperNounExtractorConfig, semantic_embedding_base_path: Path):
        super().__init__()
        self.config = config
        self.semantic_embedding_base_path = Path(semantic_embedding_base_path).expanduser()
        self.files_loaded = False
        self.proper_noun_embeddings = None  # torch.Tensor of shape (num_nouns, embedding_dim)
        self.proper_noun_to_idx = None  # dict: proper_noun_str -> index
        self.panorama_proper_nouns = None  # dict: pano_id -> list of (landmark_idx, proper_nouns, yaw_angles)

    def load_files(self):
        """Load proper noun embeddings from v2.0 pickle format."""
        base_path = self.semantic_embedding_base_path / self.config.embedding_version

        # Load proper noun embeddings
        self.proper_noun_embeddings, self.proper_noun_to_idx = load_embedding_type_across_cities(
            base_path, "proper_noun_embeddings", "proper_noun_to_idx",
            self.config.openai_embedding_size)

        # Extract panorama proper nouns mapping
        def extract_proper_nouns(pano_id_clean, pano_data):
            landmarks = []
            for lm in pano_data["landmarks"]:
                proper_nouns = lm.get("proper_nouns", [])
                if not proper_nouns:
                    continue
                landmarks.append({
                    "landmark_idx": lm["landmark_idx"],
                    "proper_nouns": proper_nouns,
                    "yaw_angles": _extract_yaw_angles_from_bboxes(lm.get("bounding_boxes", []))
                })
            return landmarks if landmarks else None

        self.panorama_proper_nouns = extract_panorama_data_across_cities(
            base_path, extract_proper_nouns)

        self.files_loaded = True
        print(f"Total proper noun embeddings loaded: {len(self.proper_noun_to_idx)}")
        print(f"Total panoramas with proper nouns: {len(self.panorama_proper_nouns)}")

    def forward(self, model_input: ModelInput) -> ExtractorOutput:
        """Extract proper noun embeddings for each panorama."""
        if not self.files_loaded:
            self.load_files()

        batch_size = len(model_input.metadata)

        # Validate panorama data
        for item in model_input.metadata:
            if 'pano_id' not in item:
                raise ValueError(
                    "PanoramaProperNounExtractor requires panorama data with 'pano_id' field.")

        # Count max proper nouns across batch
        max_proper_nouns = 0
        for item in model_input.metadata:
            pano_id = item['pano_id']
            if pano_id not in self.panorama_proper_nouns:
                continue
            count = sum(len(lm["proper_nouns"]) for lm in self.panorama_proper_nouns[pano_id])
            max_proper_nouns = max(max_proper_nouns, count)

        if max_proper_nouns == 0:
            # No proper nouns in this batch
            return ExtractorOutput(
                features=torch.zeros((batch_size, 0, self.config.openai_embedding_size),
                                    device=model_input.image.device),
                mask=torch.ones((batch_size, 0), dtype=torch.bool, device=model_input.image.device),
                positions=torch.zeros((batch_size, 0, 2, 2), device=model_input.image.device),
                debug={})

        # Initialize output tensors
        mask = torch.ones((batch_size, max_proper_nouns), dtype=torch.bool)
        features = torch.zeros((batch_size, max_proper_nouns, self.config.openai_embedding_size))
        positions = torch.zeros((batch_size, max_proper_nouns, 2, 2))

        # Process each batch item
        for i, item in enumerate(model_input.metadata):
            pano_id = item['pano_id']
            if pano_id not in self.panorama_proper_nouns:
                continue

            token_idx = 0
            for landmark_data in self.panorama_proper_nouns[pano_id]:
                yaw_angles = landmark_data["yaw_angles"]
                yaw_vector = yaw_angles_to_binary_vector(yaw_angles) if yaw_angles else [0.0, 0.0, 0.0, 0.0]

                for noun in landmark_data["proper_nouns"]:
                    if noun not in self.proper_noun_to_idx:
                        continue

                    idx = self.proper_noun_to_idx[noun]
                    embedding = self.proper_noun_embeddings[idx][:self.config.openai_embedding_size]
                    features[i, token_idx, :] = embedding

                    # Position: same yaw as parent landmark
                    positions[i, token_idx, 0, :] = torch.tensor([yaw_vector[0], yaw_vector[1]])
                    positions[i, token_idx, 1, :] = torch.tensor([yaw_vector[2], yaw_vector[3]])

                    mask[i, token_idx] = False
                    token_idx += 1

        # Re-normalize embeddings if we cropped them
        features = _normalize_cropped_embeddings(features, mask)

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
        return []


class PanoramaLocationTypeExtractor(torch.nn.Module):
    """
    Extractor for location type embeddings from panoramas.

    Location type is a scene classification like "urban commercial district".
    Produces exactly one token per panorama.
    """

    def __init__(self, config: PanoramaLocationTypeExtractorConfig, semantic_embedding_base_path: Path):
        super().__init__()
        self.config = config
        self.semantic_embedding_base_path = Path(semantic_embedding_base_path).expanduser()
        self.files_loaded = False
        self.location_type_embeddings = None  # torch.Tensor of shape (num_types, embedding_dim)
        self.location_type_to_idx = None  # dict: location_type_str -> index
        self.panorama_location_types = None  # dict: pano_id -> location_type_str

    def load_files(self):
        """Load location type embeddings from v2.0 pickle format."""
        base_path = self.semantic_embedding_base_path / self.config.embedding_version

        # Load location type embeddings (dedupe_keys=True since same types appear across cities)
        self.location_type_embeddings, self.location_type_to_idx = load_embedding_type_across_cities(
            base_path, "location_type_embeddings", "location_type_to_idx",
            self.config.openai_embedding_size, dedupe_keys=True)

        # Extract panorama location types mapping
        def extract_location_type(pano_id_clean, pano_data):
            location_type = pano_data.get("location_type", "")
            return location_type if location_type else None

        self.panorama_location_types = extract_panorama_data_across_cities(
            base_path, extract_location_type)

        self.files_loaded = True
        print(f"Total location type embeddings loaded: {len(self.location_type_to_idx)}")
        print(f"Total panoramas with location types: {len(self.panorama_location_types)}")

    def forward(self, model_input: ModelInput) -> ExtractorOutput:
        """Extract location type embedding for each panorama (one per panorama)."""
        if not self.files_loaded:
            self.load_files()

        batch_size = len(model_input.metadata)

        # Validate panorama data
        for item in model_input.metadata:
            if 'pano_id' not in item:
                raise ValueError(
                    "PanoramaLocationTypeExtractor requires panorama data with 'pano_id' field.")

        # Always exactly 1 token per panorama
        max_tokens = 1

        # Initialize output tensors
        mask = torch.ones((batch_size, max_tokens), dtype=torch.bool)
        features = torch.zeros((batch_size, max_tokens, self.config.openai_embedding_size))
        positions = torch.zeros((batch_size, max_tokens, 2, 2))  # Position (0,0) for global descriptor

        # Process each batch item
        for i, item in enumerate(model_input.metadata):
            pano_id = item['pano_id']
            if pano_id not in self.panorama_location_types:
                continue

            location_type = self.panorama_location_types[pano_id]
            if location_type not in self.location_type_to_idx:
                continue

            idx = self.location_type_to_idx[location_type]
            embedding = self.location_type_embeddings[idx][:self.config.openai_embedding_size]
            features[i, 0, :] = embedding
            mask[i, 0] = False
            # Position stays at zero (global scene descriptor)

        # Re-normalize embeddings if we cropped them
        features = _normalize_cropped_embeddings(features, mask)

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
        return []
