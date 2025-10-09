"""
Spectral Landmark Extractor for unsupervised object detection.

Detects objects using spectral decomposition of DINOv3 features and
outputs them as landmarks compatible with the existing extractor interface.
"""

import common.torch.load_torch_deps
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING

from experimental.overhead_matching.swag.model.swag_model_input_output import (
    ModelInput, ExtractorOutput
)
from experimental.overhead_matching.swag.model.spectral.dino_feature_extractor import (
    DinoFeatureHookExtractor
)
from experimental.overhead_matching.swag.model.spectral.affinity_matrix import build_affinity_matrix
from experimental.overhead_matching.swag.model.spectral.spectral_decomposer import SpectralDecomposer
from experimental.overhead_matching.swag.model.spectral.object_detector import detect_objects_from_eigenvectors

if TYPE_CHECKING:
    from experimental.overhead_matching.swag.model.swag_config_types import (
        SpectralLandmarkExtractorConfig
    )


class SpectralLandmarkExtractor(nn.Module):
    """
    Extracts object landmarks using spectral decomposition of DINO features.

    This extractor:
    1. Extracts DINO features from images
    2. Builds affinity matrix (features + optional color)
    3. Computes eigendecomposition of graph Laplacian
    4. Detects objects from eigenvectors
    5. Returns landmarks in standard ExtractorOutput format
    """

    SUPPORTED_MODELS = [
        'dinov3_vits16',
        'dinov3_vitb16',
        'dinov3_vitl16',
        'dinov3_vith16plus',
        'dinov3_vit7b16',
    ]

    def __init__(self, config: "SpectralLandmarkExtractorConfig"):
        super().__init__()

        # Validate model
        if config.dino_model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"SpectralLandmarkExtractor only supports DINOv3 ViT models. "
                f"Got: {config.dino_model}. "
                f"Supported models: {self.SUPPORTED_MODELS}"
            )

        self.config = config

        # Load DINOv3 model
        self._dino = torch.hub.load('facebookresearch/dinov3', config.dino_model)
        self._dino.eval()
        for param in self._dino.parameters():
            param.requires_grad = False

        # Setup feature extractor with hooks
        self._feature_extractor = DinoFeatureHookExtractor(
            self._dino,
            feature_source=config.feature_source
        )

        # Spectral decomposer
        self._spectral_decomposer = SpectralDecomposer(
            num_eigenvectors=config.max_landmarks_per_image
        )

        # Feature projection (if output dim differs from DINO dim)
        dino_dim = self._dino.embed_dim
        if config.output_feature_dim != dino_dim:
            self._feature_projection = nn.Linear(dino_dim, config.output_feature_dim)
        else:
            self._feature_projection = nn.Identity()

    @property
    def output_dim(self):
        return self.config.output_feature_dim

    @property
    def patch_size(self):
        return self._dino.patch_size

    def forward(self, model_input: ModelInput) -> ExtractorOutput:
        """
        Extract landmark features from images using spectral decomposition.

        Args:
            model_input: Contains images [B, 3, H, W] and metadata

        Returns:
            ExtractorOutput with:
                features: [B, max_landmarks, D]
                positions: [B, max_landmarks, 2] (relative to center)
                mask: [B, max_landmarks] (True = padding)
                debug: Dict with bboxes, eigenvectors, etc.
        """
        images = model_input.image
        batch_size = images.shape[0]
        device = images.device

        # Process each image in the batch
        all_landmarks = []
        all_eigenvectors_list = []
        all_eigenvalues_list = []

        for i in range(batch_size):
            landmarks = self._process_single_image(images[i])
            all_landmarks.append(landmarks)

        # Pack into batch format
        return self._pack_landmarks_to_output(
            all_landmarks,
            model_input,
            device
        )

    def _process_single_image(self, image: torch.Tensor) -> dict:
        """
        Process a single image through the spectral pipeline.

        Args:
            image: [3, H, W] single image

        Returns:
            Dict containing detected landmarks and debug info
        """
        # Extract DINO features [1, H_patch, W_patch, D]
        features = self._feature_extractor.extract_features(
            image.unsqueeze(0),
            normalize=True
        )[0]  # [H_patch, W_patch, D]

        H_patch, W_patch, D = features.shape

        # Build affinity matrix
        W_affinity = build_affinity_matrix(
            features,
            image=image if self.config.lambda_knn > 0 else None,
            lambda_knn=self.config.lambda_knn,
            knn_neighbors=self.config.knn_neighbors,
            intermediate_resolution=self.config.intermediate_resolution
        )

        # Compute eigendecomposition
        eigenvalues, eigenvectors = self._spectral_decomposer(W_affinity)

        # Detect objects from eigenvectors
        detected_objects = detect_objects_from_eigenvectors(
            eigenvectors,
            eigenvalues,
            features,
            min_bbox_size=self.config.min_bbox_size,
            max_objects=self.config.max_landmarks_per_image,
            aggregation_method=self.config.aggregation_method
        )

        # Reshape eigenvectors for storage [K, H, W]
        K = eigenvectors.shape[1]
        eigenvectors_spatial = eigenvectors.T.reshape(K, H_patch, W_patch)

        return {
            'objects': detected_objects,
            'eigenvectors': eigenvectors_spatial,
            'eigenvalues': eigenvalues,
            'image_shape': (image.shape[1], image.shape[2]),  # (H, W)
            'patch_shape': (H_patch, W_patch)
        }

    def _pack_landmarks_to_output(
        self,
        all_landmarks: list[dict],
        model_input: ModelInput,
        device: torch.device
    ) -> ExtractorOutput:
        """
        Pack detected landmarks into ExtractorOutput format.

        Args:
            all_landmarks: List of landmark dicts (one per image)
            model_input: Original input
            device: Target device

        Returns:
            ExtractorOutput with padded tensors
        """
        batch_size = len(all_landmarks)
        max_landmarks = self.config.max_landmarks_per_image
        output_dim = self.config.output_feature_dim

        # Initialize output tensors
        features = torch.zeros(
            (batch_size, max_landmarks, output_dim),
            dtype=torch.float32,
            device=device
        )
        positions = torch.zeros(
            (batch_size, max_landmarks, 2),
            dtype=torch.float32,
            device=device
        )
        mask = torch.ones(
            (batch_size, max_landmarks),
            dtype=torch.bool,
            device=device
        )

        # Debug tensors
        bboxes = torch.zeros(
            (batch_size, max_landmarks, 4),
            dtype=torch.float32,
            device=device
        )
        eigenvector_indices = torch.zeros(
            (batch_size, max_landmarks),
            dtype=torch.long,
            device=device
        )
        eigenvalues_out = torch.zeros(
            (batch_size, max_landmarks),
            dtype=torch.float32,
            device=device
        )

        # Store eigenvectors and masks (variable size, stored as list in debug)
        all_eigenvectors = []
        all_object_masks = []

        # Fill tensors
        for i, landmark_dict in enumerate(all_landmarks):
            objects = landmark_dict['objects']
            num_objects = len(objects)

            if num_objects == 0:
                # No objects detected, everything stays as padding
                all_eigenvectors.append(landmark_dict['eigenvectors'])
                all_object_masks.append(torch.zeros(
                    (0, *landmark_dict['patch_shape']),
                    device=device
                ))
                continue

            # Get image center for relative positioning
            H_img, W_img = landmark_dict['image_shape']
            center_y, center_x = H_img // 2, W_img // 2

            # Fill each detected object
            for j, obj in enumerate(objects[:max_landmarks]):
                # Project feature
                feature_projected = self._feature_projection(obj.feature)
                features[i, j] = feature_projected

                # Compute bbox center relative to image center
                x1, y1, x2, y2 = obj.bbox
                bbox_center_y = (y1 + y2) / 2
                bbox_center_x = (x1 + x2) / 2

                # Scale from patch coordinates to image coordinates
                patch_size = self.patch_size
                bbox_center_y_img = bbox_center_y * patch_size
                bbox_center_x_img = bbox_center_x * patch_size

                # Relative position (dy, dx) from center
                positions[i, j, 0] = bbox_center_y_img - center_y
                positions[i, j, 1] = bbox_center_x_img - center_x

                # Unmask this landmark
                mask[i, j] = False

                # Store debug info
                bboxes[i, j] = torch.tensor(
                    [x1 * patch_size, y1 * patch_size,
                     x2 * patch_size, y2 * patch_size],
                    dtype=torch.float32,
                    device=device
                )
                eigenvector_indices[i, j] = obj.eigenvector_idx
                eigenvalues_out[i, j] = obj.eigenvalue

            # Store eigenvectors and masks
            all_eigenvectors.append(landmark_dict['eigenvectors'])
            object_masks = torch.stack([obj.soft_mask for obj in objects], dim=0)
            all_object_masks.append(object_masks)

        # Create debug dict
        debug = {
            'bboxes': bboxes,
            'eigenvector_indices': eigenvector_indices,
            'eigenvalues': eigenvalues_out,
            'eigenvectors': all_eigenvectors,  # List of [K, H, W] tensors
            'object_masks': all_object_masks,  # List of [N_obj, H, W] tensors
        }

        return ExtractorOutput(
            features=features,
            positions=positions,
            mask=mask,
            debug=debug
        )
