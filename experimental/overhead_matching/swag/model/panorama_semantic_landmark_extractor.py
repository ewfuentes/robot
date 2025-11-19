
import common.torch.load_torch_deps
import torch
import math
import json
import base64
import pickle
from pathlib import Path
from experimental.overhead_matching.swag.model.swag_config_types import (
    PanoramaSemanticLandmarkExtractorConfig, ExtractorDataRequirement)
from experimental.overhead_matching.swag.model.swag_model_input_output import (
    ModelInput, ExtractorOutput)
from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    load_all_jsonl_from_folder, make_embedding_dict_from_json, make_sentence_dict_from_pano_jsons)
import base64
from typing import NamedTuple


class GroupClassification(NamedTuple):
    low_level_similarities: torch.Tensor
    low_level_classification: torch.Tensor
    high_level_classification: torch.Tensor


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


def classify_against_grouping(features, semantic_grouping) -> GroupClassification:
    all_low_level_classes = list(semantic_grouping["class_details"].keys())
    num_low_level_classes = len(all_low_level_classes)
    num_high_level_classes = len(semantic_grouping["semantic_groups"])

    low_level_class_embeddings = []
    for v in semantic_grouping["class_details"].values():
        low_level_class_embeddings.append(v["embedding"]["vector"])
    low_level_class_embeddings = torch.stack(low_level_class_embeddings)

    # Create a lookup table from low level classes to high level classes
    high_level_class_from_low_level = torch.zeros((num_high_level_classes, num_low_level_classes))
    for hlc_idx, low_level_classes in enumerate(semantic_grouping["semantic_groups"].values()):
        for llc in low_level_classes:
            llc_idx = all_low_level_classes.index(llc)
            high_level_class_from_low_level[hlc_idx, llc_idx] = 1.0

    similarities = features @ low_level_class_embeddings.T
    max_similarities = torch.argmax(similarities, -1)
    max_one_hot = torch.nn.functional.one_hot(max_similarities, num_classes=num_low_level_classes)
    out = max_one_hot.to(torch.float32) @ high_level_class_from_low_level.T
    return GroupClassification(
        low_level_similarities=similarities,
        low_level_classification=max_one_hot,
        high_level_classification=out
    )


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

        # Load the semantic class groupings (only if needed)
        self.semantic_groupings = None
        if self.config.should_classify_against_grouping:
            base_path = self.semantic_embedding_base_path / self.config.embedding_version
            semantic_groupings_file = base_path / "semantic_class_grouping.json"
            if not semantic_groupings_file.exists():
                raise FileNotFoundError(
                    f"Semantic groupings file not found: {semantic_groupings_file}. "
                    f"Either provide this file or set should_classify_against_grouping=False in config.")
            self.semantic_groupings = json.loads(semantic_groupings_file.read_text())

            # Convert the base64 encoded embeddings into torch tensors
            for k, v in self.semantic_groupings["class_details"].items():
                base64_string = v["embedding"]["vector"]
                base64_buffer = bytearray(base64.b64decode(base64_string))
                v["embedding"]["vector"] = torch.frombuffer(base64_buffer, dtype=torch.float32)

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

            # Load embeddings
            embedding_dir = city_dir / "embeddings"
            if embedding_dir.exists():
                # Try to load from pickle file first (faster)
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
            city_sentences, metadata_from_sentences, _ = make_sentence_dict_from_pano_jsons(
                load_all_jsonl_from_folder(sentence_dir))
            self.all_sentences.update(city_sentences)
            print(f"  Loaded {len(city_sentences)} sentences")

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
                if metadata_from_sentences is not None and len(metadata_from_sentences) > 0:
                    assert metadata_from_sentences == new_metadata
                self.panorama_metadata.update(new_metadata)
                assert len(self.panorama_metadata) == old_metadata_size + new_pano_metadata_len
            elif metadata_from_sentences is not None:
                # If no metadata file exists, use the metadata derived from sentences
                old_metadata_size = len(self.panorama_metadata)
                new_pano_metadata_len = len(metadata_from_sentences)
                print(f"  Loaded metadata for {len(metadata_from_sentences)} panoramas (from sentences)")
                self.panorama_metadata.update(metadata_from_sentences)
                assert len(self.panorama_metadata) == old_metadata_size + new_pano_metadata_len

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
                print(f"Failed to find pano id {pano_id} in panorama metadata! Skipping landmark")
                continue
            matching_landmarks = self.panorama_metadata[pano_id]

            # Sort by landmark_idx to ensure consistent ordering
            matching_landmarks = sorted(matching_landmarks, key=lambda x: x["landmark_idx"])

            for landmark_idx, landmark_meta in enumerate(matching_landmarks):
                custom_id = landmark_meta["custom_id"]
                yaw_angles = landmark_meta["yaw_angles"]

                # Get embedding
                if custom_id not in self.all_embeddings:
                    print(f"Warning: missing embedding for {custom_id}")
                    continue

                # Crop embedding if needed
                features[i, landmark_idx, :] = torch.tensor(
                    self.all_embeddings[custom_id])[:self.config.openai_embedding_size]

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
                if custom_id in self.all_sentences:
                    max_description_length = max(
                        max_description_length,
                        len(self.all_sentences[custom_id].encode("utf-8")))

        debug = {}
        if self.config.should_classify_against_grouping:
            groupings = classify_against_grouping(features, self.semantic_groupings)
            debug["low_level_similarity"] = groupings.low_level_similarities.to(model_input.image.device)
            debug["low_level_classification"] = groupings.low_level_classification.to(model_input.image.device)
            features = groupings.high_level_classification
        else:
            # Re-normalize embeddings if we cropped them
            features[~mask] = features[~mask] / torch.norm(features[~mask], dim=-1).unsqueeze(-1)

        # Create debug tensor for sentences
        sentence_debug = torch.zeros(
            (batch_size, max_num_landmarks, max_description_length), dtype=torch.uint8)

        for i, item in enumerate(model_input.metadata):
            pano_id = item['pano_id']
            if pano_id not in self.panorama_metadata:
                continue

            # Find matching panorama metadata
            matching_landmarks = self.panorama_metadata[pano_id]

            matching_landmarks = sorted(matching_landmarks, key=lambda x: x["landmark_idx"])

            for landmark_idx, landmark_meta in enumerate(matching_landmarks):
                custom_id = landmark_meta["custom_id"]
                if custom_id in self.all_sentences:
                    sentence_bytes = self.all_sentences[custom_id].encode('utf-8')
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
        out = self.config.openai_embedding_size
        if self.config.should_classify_against_grouping:
            out = len(self.semantic_groupings["semantic_groups"])
        return out

    @property
    def num_position_outputs(self):
        return 2

    @property
    def data_requirements(self) -> list[ExtractorDataRequirement]:
        # This extractor doesn't use landmarks from the dataset
        # (it uses vision-extracted landmarks stored separately)
        return []
