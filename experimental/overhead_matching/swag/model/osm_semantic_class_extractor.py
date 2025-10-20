
import common.torch.load_torch_deps
import torch
import json
from pathlib import Path
from typing import Optional

from experimental.overhead_matching.swag.model.swag_config_types import (
    OSMSemanticClassExtractorConfig, ExtractorDataRequirement)
from experimental.overhead_matching.swag.model.swag_model_input_output import (
    ModelInput, ExtractorOutput)
from experimental.overhead_matching.swag.model.semantic_landmark_extractor import (
    compute_landmark_pano_positions, compute_landmark_sat_positions)
from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    prune_landmark
)


class OSMSemanticClassExtractor(torch.nn.Module):
    """
    Extracts embeddings for OSM landmarks based on their semantic class.

    Maps OSM tag combinations to broad semantic classes and returns one-hot
    embeddings based on the class index.
    """

    def __init__(self, config: OSMSemanticClassExtractorConfig, base_path: str):
        super().__init__()
        self.config = config

        # Construct full path from base_path and embedding_version
        json_filename = f"{config.embedding_version}.json"
        json_path = Path(base_path).expanduser() / json_filename

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract semantic groups (broad classes) and create ontology
        # The ontology is a sorted list of broad class names for consistent indexing
        semantic_groups = data['semantic_groups']
        self.ontology = sorted(semantic_groups.keys())
        self.num_classes = len(self.ontology)
        self.broad_class_to_id = {cls: idx for idx, cls in enumerate(self.ontology)}

        # Build reverse mapping: semantic_class_name -> broad_class
        semantic_class_to_broad = {}
        for broad_class, semantic_class_list in semantic_groups.items():
            for semantic_class in semantic_class_list:
                semantic_class_to_broad[semantic_class] = broad_class

        # Load class details and build mappings: tags -> semantic_class -> broad_class
        class_details = data['class_details']
        self.mappings = []

        for semantic_class_name, details in class_details.items():
            osm_tags = details['osm_tags']

            # Convert OSM tags dict to frozenset of (key, value) tuples
            tags_frozen = frozenset(osm_tags.items())

            # Look up which broad class this semantic class belongs to
            if semantic_class_name not in semantic_class_to_broad:
                print(f"Warning: Semantic class '{semantic_class_name}' not found in any semantic group")
                continue

            broad_class = semantic_class_to_broad[semantic_class_name]

            self.mappings.append({
                'tags': tags_frozen,
                'num_tags': len(tags_frozen),
                'broad_class': broad_class,
                'semantic_class': semantic_class_name
            })

        # Sort by number of tags (descending) for most-specific matching
        self.mappings.sort(key=lambda x: x['num_tags'], reverse=True)

        print(f"OSMSemanticClassExtractor: Loaded {len(self.mappings)} mappings "
              f"across {self.num_classes} broad semantic classes")

    def map_tags_to_class_id(self, tag_frozenset: frozenset) -> Optional[int]:
        """
        Map OSM tags to a semantic class ID.

        Uses most-specific match: finds the mapping with the largest subset
        of tags present in the input. Relies on self.mappings being sorted

        Args:
            tag_frozenset: frozenset of (key, value) tuples

        Returns:
            Class ID (0-indexed) or None if no match
        """
        for mapping in self.mappings:
            if mapping['tags'].issubset(tag_frozenset):
                broad_class = mapping['broad_class']
                return self.broad_class_to_id[broad_class]
        return None

    @property
    def output_dim(self) -> int:
        """Output embedding dimension equals number of classes."""
        return self.num_classes

    @property
    def num_position_outputs(self) -> int:
        """Number of position outputs (satellite, panorama bearing)."""
        return 2

    @property
    def data_requirements(self) -> list[ExtractorDataRequirement]:
        """This extractor needs landmark data."""
        return [ExtractorDataRequirement.LANDMARKS]

    def forward(self, model_input: ModelInput) -> ExtractorOutput:
        """
        Extract semantic class embeddings for landmarks.

        Args:
            model_input: Input containing landmark metadata

        Returns:
            ExtractorOutput with one-hot embeddings based on semantic class
        """
        batch_size = len(model_input.metadata)

        # Check if we're processing panorama or satellite images
        is_panorama = 'pano_id' in model_input.metadata[0]

        # Get landmarks from metadata
        all_landmarks = []
        max_num_landmarks = 0

        for batch_idx in range(batch_size):
            metadata = model_input.metadata[batch_idx]
            landmarks = metadata['landmarks']
            all_landmarks.append(landmarks)
            max_num_landmarks = max(max_num_landmarks, len(landmarks))

        # Initialize output tensors
        mask = torch.ones((batch_size, max_num_landmarks), dtype=torch.bool)
        features = torch.zeros((batch_size, max_num_landmarks, self.output_dim))
        positions = torch.zeros((batch_size, max_num_landmarks, self.num_position_outputs, 2))

        # Process each batch item
        for batch_idx in range(batch_size):
            item = model_input.metadata[batch_idx]
            landmarks = all_landmarks[batch_idx]
            num_landmarks = len(landmarks)

            if num_landmarks == 0:
                continue

            # Compute positions using the same method as SemanticLandmarkExtractor
            if is_panorama:
                positions[batch_idx, :num_landmarks] = compute_landmark_pano_positions(
                    item, model_input.image.shape[-2:])
            else:
                positions[batch_idx, :num_landmarks] = compute_landmark_sat_positions(item)


            for landmark_idx, landmark in enumerate(landmarks):
                # Convert properties to frozenset of (key, value) tuples
                tag_set = prune_landmark(landmark)

                # Map to class ID
                class_id = self.map_tags_to_class_id(tag_set)

                if class_id is not None:
                    # One-hot encoding
                    features[batch_idx, landmark_idx, class_id] = 1.0
                else:
                    # Unmatched landmarks get all zeros
                    features[batch_idx, landmark_idx, :] = 0.0

            mask[batch_idx, :num_landmarks] = False


        device = model_input.image.device

        return ExtractorOutput(
            mask=mask.to(device),
            features=features.to(device),
            positions=positions.to(device)
        )
