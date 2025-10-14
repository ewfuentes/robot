import common.torch.load_torch_deps
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any

from experimental.overhead_matching.swag.model.swag_patch_embedding import DinoFeatureExtractor
from experimental.overhead_matching.swag.model.swag_config_types import DinoFeatureMapExtractorConfig
from experimental.overhead_matching.swag.model.swag_model_input_output import ModelInput


class ImageSimilarityEngine:
    def __init__(self, model_str: str = "dinov2_vitb14", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Use existing DINO infrastructure - it should handle both DINOv2 and DINOv3
        config = DinoFeatureMapExtractorConfig(
            model_str=model_str,
            use_class_token_only=False
        )
        self.extractor = DinoFeatureExtractor(config).to(self.device)
        self.patch_size = self.extractor.patch_size[0]  # Assuming square patches

        # Store extracted features for loaded images
        self.image_features = {}
        self.image_positions = {}
        self.image_shapes = {}
        self.image_metadata = {}

    def load_image(self, image_input: Union[str, Dict[str, Any]]) -> tuple[torch.Tensor, Image.Image, Dict[str, Any]]:
        """Load and preprocess image for DINO feature extraction.

        Args:
            image_input: Either a file path string or a dict with VIGOR data
                        Dict format: {'array': np.ndarray, 'metadata': dict, 'type': str}

        Returns:
            (image_tensor, pil_image, metadata)
        """
        if isinstance(image_input, str):
            # File path input
            image = Image.open(image_input).convert('RGB')
            metadata = {'source': 'file', 'path': image_input}
        elif isinstance(image_input, dict):
            # VIGOR dataset input
            img_array = image_input['array']
            metadata = image_input.get('metadata', {})
            metadata['source'] = 'vigor'
            metadata['type'] = image_input.get('type', 'unknown')

            # Convert numpy array to PIL Image
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)

            image = Image.fromarray(img_array)
        else:
            raise ValueError("image_input must be either a file path string or a VIGOR data dict")

        # Convert to tensor and add batch dimension
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        image_tensor = transform(image).unsqueeze(0).to(self.device)
        return image_tensor, image, metadata

    def extract_features(self, image_input: Union[str, Dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract DINO features for an image and cache them."""
        # Create a unique identifier for caching
        if isinstance(image_input, str):
            image_id = image_input
        else:
            image_id = f"{image_input.get('type', 'unknown')}_{image_input.get('dataset_index', 0)}"

        if image_id in self.image_features:
            return self.image_features[image_id], self.image_positions[image_id]

        image_tensor, pil_image, metadata = self.load_image(image_input)

        # Create ModelInput in the format expected by DinoFeatureExtractor
        model_input = ModelInput(
            image=image_tensor,
            metadata=[metadata],  # Include metadata
            cached_tensors={}
        )

        # Extract features using existing DINO infrastructure (handles both DINOv2 and DINOv3)
        with torch.no_grad():
            output = self.extractor(model_input)

        features = output.features.squeeze(0)  # Remove batch dimension
        positions = output.positions.squeeze(0)  # Remove batch dimension

        # Cache the results
        self.image_features[image_id] = features
        self.image_positions[image_id] = positions
        self.image_shapes[image_id] = pil_image.size  # (width, height)
        self.image_metadata[image_id] = metadata

        return features, positions

    def pixel_to_patch_idx(self, pixel_x: int, pixel_y: int, image_id: str) -> int:
        """Convert pixel coordinates to patch index."""
        if image_id not in self.image_shapes:
            raise ValueError(f"Image {image_id} not loaded")

        width, height = self.image_shapes[image_id]

        # Calculate patch grid dimensions
        patches_per_row = height // self.patch_size
        patches_per_col = width // self.patch_size

        # Convert pixel coordinates to patch coordinates
        patch_row = min(pixel_y // self.patch_size, patches_per_row - 1)
        patch_col = min(pixel_x // self.patch_size, patches_per_col - 1)

        # Convert to linear patch index
        patch_idx = patch_row * patches_per_col + patch_col

        return patch_idx

    def compute_similarities(self, clicked_image_input: Union[str, Dict[str, Any]], clicked_x: int, clicked_y: int,
                           target_image_inputs: list[Union[str, Dict[str, Any]]]) -> dict[str, np.ndarray]:
        """Compute similarity maps for clicked point across all target images."""
        # Create image ID for clicked image
        if isinstance(clicked_image_input, str):
            clicked_image_id = clicked_image_input
        else:
            clicked_image_id = f"{clicked_image_input.get('type', 'unknown')}_{clicked_image_input.get('dataset_index', 0)}"

        # Get features for clicked image
        clicked_features, _ = self.extract_features(clicked_image_input)

        # Get the patch index for the clicked point
        clicked_patch_idx = self.pixel_to_patch_idx(clicked_x, clicked_y, clicked_image_id)

        if clicked_patch_idx >= clicked_features.shape[0]:
            raise ValueError(f"Click outside valid patch area")

        # Get the feature vector for the clicked patch
        clicked_patch_feature = clicked_features[clicked_patch_idx]

        similarities = {}

        for target_input in target_image_inputs:
            # Create image ID for target
            if isinstance(target_input, str):
                target_id = target_input
            else:
                target_id = f"{target_input.get('type', 'unknown')}_{target_input.get('dataset_index', 0)}"

            # Extract features for target image
            target_features, target_positions = self.extract_features(target_input)

            # Compute cosine similarity between clicked patch and all target patches
            similarity_scores = F.cosine_similarity(
                clicked_patch_feature.unsqueeze(0),
                target_features,
                dim=1
            )

            # Convert to numpy for visualization
            similarities[target_id] = similarity_scores.cpu().numpy()

        return similarities

    def similarity_to_heatmap(self, similarities: np.ndarray, image_id: str) -> np.ndarray:
        """Convert patch-wise similarities to a spatial heatmap."""
        if image_id not in self.image_shapes:
            raise ValueError(f"Image {image_id} not loaded")

        width, height = self.image_shapes[image_id]
        patches_per_row = height // self.patch_size
        patches_per_col = width // self.patch_size

        # Reshape similarities to spatial grid
        heatmap = similarities[:patches_per_row * patches_per_col].reshape(
            patches_per_row, patches_per_col
        )

        # Upsample to image resolution for overlay
        heatmap_upsampled = np.repeat(np.repeat(heatmap, self.patch_size, axis=0),
                                     self.patch_size, axis=1)

        # Crop to exact image dimensions if needed
        heatmap_upsampled = heatmap_upsampled[:height, :width]

        return heatmap_upsampled

    def get_image_metadata(self, image_id: str) -> Dict[str, Any]:
        """Get metadata for a loaded image."""
        return self.image_metadata.get(image_id, {})
