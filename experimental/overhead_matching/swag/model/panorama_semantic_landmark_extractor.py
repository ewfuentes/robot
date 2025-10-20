
import common.torch.load_torch_deps
import torch
import math
import json
from pathlib import Path
from experimental.overhead_matching.swag.model.swag_config_types import (
    PanoramaSemanticLandmarkExtractorConfig, ExtractorDataRequirement)
from experimental.overhead_matching.swag.model.swag_model_input_output import (
    ModelInput, ExtractorOutput)
from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    load_all_jsonl_from_folder, make_embedding_dict_from_json)


def make_sentence_dict_from_json(sentence_jsons: list) -> tuple[dict[str, str], int]:
    """Create a dictionary mapping custom_id to sentence description."""
    out = {}
    output_tokens = 0
    for response in sentence_jsons:
        if len(response['response']['body']) == 0:
            print(f"GOT EMPTY RESPONSE {response}. SKIPPING")
            continue
        assert response["error"] == None and \
            response["response"]["body"]["choices"][0]["finish_reason"] == "stop" and \
            response["response"]["body"]["choices"][0]["message"]["refusal"] == None
        custom_id = response["custom_id"]

        # Parse the JSON content to extract landmarks
        content_str = response["response"]["body"]["choices"][0]["message"]["content"]
        try:
            content = json.loads(content_str)
            landmarks = content.get("landmarks", [])

            # Create entries for each landmark in this panorama
            panorama_id = response["custom_id"]
            for idx, landmark in enumerate(landmarks):
                description = landmark.get("description", "")
                if description:
                    landmark_custom_id = f"{panorama_id}__landmark_{idx}"
                    out[landmark_custom_id] = description
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON for {custom_id}: {e}")
            continue

        output_tokens += response["response"]["body"]["usage"]["completion_tokens"]
    return out, output_tokens


def yaw_angles_to_radians(yaw_degrees: list[int]) -> tuple[float, float]:
    """
    Convert yaw angles in degrees to angular bounds in radians.

    Panorama coordinate system:
    - 0° yaw = north = 0 radians (center of panorama)
    - 90° yaw = east = π/2 radians (CCW from north)
    - 180° yaw = south = π radians (edges)
    - 270° yaw = west = -π/2 radians (CW from north)

    For continuous ranges (adjacent angles like [0, 90] or [270, 0]),
    returns (min_angle, max_angle).

    For discontinuous ranges (non-adjacent like [90, 270] or [0, 180]),
    returns (first_angle, first_angle) to use just the first value.

    Returns:
        (angle1, angle2) in radians, range [-π, π]
    """
    if not yaw_degrees:
        return (0.0, 0.0)

    if len(yaw_degrees) == 1:
        # Single angle: return it twice (degenerate range)
        rad = math.radians(yaw_degrees[0])
        # Normalize to [-π, π]
        if rad > math.pi:
            rad -= 2 * math.pi
        return (rad, rad)

    # Convert to radians and normalize to [-π, π]
    radians = []
    for deg in yaw_degrees:
        rad = math.radians(deg)
        # Normalize: 270° = 3π/2 → -π/2
        if rad > math.pi:
            rad -= 2 * math.pi
        radians.append(rad)

    # Sort angles
    radians_sorted = sorted(radians)

    # Check if angles form a continuous range
    # Continuous means all consecutive angles are < 180° apart (strictly less than)
    # Angles exactly 180° apart or more are considered discontinuous
    # Don't check wrap-around (last to first) for continuity - only forward gaps
    is_continuous = True
    for i in range(len(radians_sorted) - 1):  # Don't wrap around
        gap = radians_sorted[i + 1] - radians_sorted[i]
        # Gap should be positive and < π for continuous range (strictly less than)
        if gap >= math.pi - 1e-6:  # Use small epsilon for floating point comparison
            is_continuous = False
            break

    if not is_continuous:
        # Discontinuous range: use only the first angle
        first_angle = radians[0]  # Use original first angle, not sorted
        return (first_angle, first_angle)

    # Continuous range: return min and max
    # Need to handle wrap-around (e.g., [270°, 0°] = [-π/2, 0])
    min_angle = radians_sorted[0]
    max_angle = radians_sorted[-1]

    return (min_angle, max_angle)


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
                city_embeddings = make_embedding_dict_from_json(
                    load_all_jsonl_from_folder(embedding_dir))
                self.all_embeddings.update(city_embeddings)
                print(f"  Loaded {len(city_embeddings)} embeddings")

            # Load sentences (optional)
            sentence_dir = city_dir / "sentences"
            if sentence_dir.exists():
                city_sentences, _ = make_sentence_dict_from_json(
                    load_all_jsonl_from_folder(sentence_dir))
                self.all_sentences.update(city_sentences)
                print(f"  Loaded {len(city_sentences)} sentences")

            # Load panorama metadata
            metadata_file = city_dir / "embedding_requests" / "panorama_metadata.jsonl"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    for line in f:
                        meta = json.loads(line)
                        pano_id = meta["panorama_id"]
                        landmark_idx = meta["landmark_idx"]
                        custom_id = meta["custom_id"]
                        yaw_angles = meta.get("yaw_angles", [])

                        if pano_id not in self.panorama_metadata:
                            self.panorama_metadata[pano_id] = []

                        self.panorama_metadata[pano_id].append({
                            "landmark_idx": landmark_idx,
                            "custom_id": custom_id,
                            "yaw_angles": yaw_angles
                        })
                print(f"  Loaded metadata for {len([k for k in self.panorama_metadata if k.startswith(city_name.split()[0])])} panoramas")

        assert len(self.all_embeddings) > 0, f"Failed to load any embeddings from {base_path}"
        assert len(next(iter(self.all_embeddings.values()))) >= self.config.openai_embedding_size, \
            f"Requested embedding length ({self.config.openai_embedding_size}) longer than available ({len(next(iter(self.all_embeddings.values())))})"

        print(f"Total embeddings loaded: {len(self.all_embeddings)}")
        print(f"Total panoramas with landmarks: {len(self.panorama_metadata)}")

    def forward(self, model_input: ModelInput) -> ExtractorOutput:
        """Extract panorama-based semantic landmark features."""
        if not self.files_loaded:
            self.load_files()
            self.files_loaded = True

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
            # The pano_id from metadata might have extra info, match by prefix
            # Find matching panorama in our metadata
            matching_landmarks = None
            for meta_pano_id in self.panorama_metadata:
                if pano_id in meta_pano_id or meta_pano_id in pano_id:
                    matching_landmarks = self.panorama_metadata[meta_pano_id]
                    break

            num_landmarks = len(matching_landmarks) if matching_landmarks else 0
            valid_landmarks.append(num_landmarks)

        max_num_landmarks = max(valid_landmarks) if valid_landmarks else 0

        # Initialize output tensors
        mask = torch.ones((batch_size, max_num_landmarks), dtype=torch.bool)
        features = torch.zeros((batch_size, max_num_landmarks, self.output_dim))
        positions = torch.zeros((batch_size, max_num_landmarks, 2, 2))

        max_description_length = 0

        # Process each batch item
        for i, item in enumerate(model_input.metadata):
            pano_id = item['pano_id']

            # Find matching panorama metadata
            matching_landmarks = None
            for meta_pano_id in self.panorama_metadata:
                if pano_id in meta_pano_id or meta_pano_id in pano_id:
                    matching_landmarks = self.panorama_metadata[meta_pano_id]
                    break

            if not matching_landmarks:
                continue

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
                    self.all_embeddings[custom_id])[:self.output_dim]

                # Compute angular positions from yaw angles
                angle1, angle2 = yaw_angles_to_radians(yaw_angles)

                # Position format: [batch, num_landmarks, 2, 2]
                # Shape is [batch, num_landmarks, 2, 2] where:
                # - dim 2 (size 2): min/max bounds of the landmark
                # - dim 3 (size 2): [horizontal_angle, horizontal_angle]
                #   (no vertical component - landmarks are at horizon level)
                # For SphericalPositionEmbedding, both positions are just horizontal angles
                positions[i, landmark_idx, 0, :] = torch.tensor([angle1, angle1])
                positions[i, landmark_idx, 1, :] = torch.tensor([angle2, angle2])

                # Mark as valid (False = not masked)
                mask[i, landmark_idx] = False

                # Track max description length for debug tensor
                if self.all_sentences and custom_id in self.all_sentences:
                    max_description_length = max(
                        max_description_length,
                        len(self.all_sentences[custom_id].encode("utf-8")))

        # Re-normalize embeddings if we cropped them
        features[~mask] = features[~mask] / torch.norm(features[~mask], dim=-1).unsqueeze(-1)

        # Create debug tensor for sentences
        sentence_debug = torch.zeros(
            (batch_size, max_num_landmarks, max_description_length), dtype=torch.uint8)

        if self.all_sentences:
            for i, item in enumerate(model_input.metadata):
                pano_id = item['pano_id']

                # Find matching panorama metadata
                matching_landmarks = None
                for meta_pano_id in self.panorama_metadata:
                    if pano_id in meta_pano_id or meta_pano_id in pano_id:
                        matching_landmarks = self.panorama_metadata[meta_pano_id]
                        break

                if not matching_landmarks:
                    continue

                matching_landmarks = sorted(matching_landmarks, key=lambda x: x["landmark_idx"])

                for landmark_idx, landmark_meta in enumerate(matching_landmarks):
                    custom_id = landmark_meta["custom_id"]
                    if custom_id in self.all_sentences:
                        sentence_bytes = self.all_sentences[custom_id].encode('utf-8')
                        sentence_tensor = torch.tensor(list(sentence_bytes), dtype=torch.uint8)
                        sentence_debug[i, landmark_idx, :len(sentence_bytes)] = sentence_tensor

        return ExtractorOutput(
            features=features.to(model_input.image.device),
            mask=mask.to(model_input.image.device),
            positions=positions.to(model_input.image.device),
            debug={'sentences': sentence_debug.to(model_input.image.device)})

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
