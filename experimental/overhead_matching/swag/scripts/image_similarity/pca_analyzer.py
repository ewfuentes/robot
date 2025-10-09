import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from pathlib import Path


class PCAAnalyzer:
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.components = None
        self.explained_variance_ratio = None
        self.mean_features = None
        self.is_fitted = False

        # Store per-image information for visualization
        self.image_pca_projections = {}
        self.image_positions = {}
        self.image_shapes = {}

    def fit(self, image_features: Dict[str, torch.Tensor],
            image_positions: Dict[str, torch.Tensor],
            image_shapes: Dict[str, Tuple[int, int]]) -> None:
        """
        Compute global PCA across all image features.

        Args:
            image_features: Dict mapping image paths to feature tensors [num_patches, feature_dim]
            image_positions: Dict mapping image paths to position tensors [num_patches, 2]
            image_shapes: Dict mapping image paths to (width, height) tuples
        """
        if not image_features:
            raise ValueError("No image features provided for PCA computation")

        # Store image metadata
        self.image_positions = image_positions
        self.image_shapes = image_shapes

        # Concatenate all patch features from all images
        all_features = []
        for image_path, features in image_features.items():
            all_features.append(features.cpu().numpy())

        combined_features = np.concatenate(all_features, axis=0)  # [total_patches, feature_dim]

        # Center the data
        self.mean_features = np.mean(combined_features, axis=0)
        centered_features = combined_features - self.mean_features

        # Compute SVD for PCA
        # For efficiency with high-dimensional data, we can use the covariance trick
        # if n_samples < n_features
        n_samples, n_features = centered_features.shape

        if n_samples < n_features:
            # Use covariance matrix approach (more efficient for high-dimensional features)
            cov_matrix = np.dot(centered_features, centered_features.T) / (n_samples - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # Sort by eigenvalues (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Transform back to feature space
            self.components = np.dot(eigenvectors[:, :self.n_components].T, centered_features)
            # Normalize components
            for i in range(self.components.shape[0]):
                self.components[i] /= np.linalg.norm(self.components[i])

            self.explained_variance_ratio = eigenvalues[:self.n_components] / np.sum(eigenvalues)
        else:
            # Standard SVD approach
            U, S, Vt = np.linalg.svd(centered_features, full_matrices=False)

            # Principal components are rows of Vt
            self.components = Vt[:self.n_components]  # [n_components, feature_dim]
            self.explained_variance_ratio = (S[:self.n_components] ** 2) / np.sum(S ** 2)

        # Project each image's features onto the principal components
        self._compute_image_projections(image_features)

        self.is_fitted = True

    def _compute_image_projections(self, image_features: Dict[str, torch.Tensor]) -> None:
        """Compute PCA projections for each image separately."""
        self.image_pca_projections = {}

        for image_path, features in image_features.items():
            features_np = features.cpu().numpy()
            centered_features = features_np - self.mean_features

            # Project onto principal components
            projections = np.dot(centered_features, self.components.T)  # [num_patches, n_components]
            self.image_pca_projections[image_path] = projections

    def get_component_heatmap(self, image_path: str, component_idx: int) -> np.ndarray:
        """
        Get the spatial heatmap for a specific principal component and image.

        Args:
            image_path: Path or identifier for the image
            component_idx: Index of the principal component (0-based)

        Returns:
            Heatmap array matching the image dimensions
        """
        if not self.is_fitted:
            raise ValueError("PCA has not been fitted yet")

        if component_idx >= self.n_components:
            raise ValueError(f"Component index {component_idx} >= n_components {self.n_components}")

        if image_path not in self.image_pca_projections:
            raise ValueError(f"Image {image_path} not found in PCA results")

        projections = self.image_pca_projections[image_path]
        component_values = projections[:, component_idx]  # [num_patches]

        # Convert to spatial heatmap
        return self._values_to_heatmap(component_values, image_path)

    def _values_to_heatmap(self, patch_values: np.ndarray, image_path: str) -> np.ndarray:
        """Convert patch-wise values to a spatial heatmap matching image dimensions."""
        if image_path not in self.image_shapes:
            raise ValueError(f"Image shape not found for {image_path}")

        width, height = self.image_shapes[image_path]
        positions = self.image_positions[image_path].cpu().numpy()

        # Determine patch grid dimensions (assuming regular grid)
        # This is a simplified approach - in practice, you might need more sophisticated mapping
        patch_size = 16  # Default DINO patch size
        patches_per_row = height // patch_size
        patches_per_col = width // patch_size

        # Reshape patch values to spatial grid
        if len(patch_values) != patches_per_row * patches_per_col:
            # Handle case where number of patches doesn't match exactly
            # This can happen due to padding or different patch arrangements
            grid_values = np.zeros(patches_per_row * patches_per_col)
            grid_values[:len(patch_values)] = patch_values
        else:
            grid_values = patch_values

        spatial_grid = grid_values.reshape(patches_per_row, patches_per_col)

        # Upsample to image resolution
        heatmap = np.repeat(np.repeat(spatial_grid, patch_size, axis=0), patch_size, axis=1)

        # Crop to exact image dimensions if needed
        heatmap = heatmap[:height, :width]

        return heatmap

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get the explained variance ratio for each component."""
        if not self.is_fitted:
            raise ValueError("PCA has not been fitted yet")
        return self.explained_variance_ratio.copy()

    def get_component_summary(self) -> List[Dict[str, Any]]:
        """Get a summary of all principal components."""
        if not self.is_fitted:
            raise ValueError("PCA has not been fitted yet")

        summary = []
        for i in range(self.n_components):
            summary.append({
                'component_idx': i,
                'explained_variance_ratio': float(self.explained_variance_ratio[i]),
                'cumulative_variance_ratio': float(np.sum(self.explained_variance_ratio[:i+1]))
            })

        return summary

    def transform_new_image_features(self, features: torch.Tensor) -> np.ndarray:
        """
        Transform new image features using the fitted PCA.

        Args:
            features: Feature tensor [num_patches, feature_dim]

        Returns:
            PCA projections [num_patches, n_components]
        """
        if not self.is_fitted:
            raise ValueError("PCA has not been fitted yet")

        features_np = features.cpu().numpy()
        centered_features = features_np - self.mean_features

        # Project onto principal components
        projections = np.dot(centered_features, self.components.T)
        return projections