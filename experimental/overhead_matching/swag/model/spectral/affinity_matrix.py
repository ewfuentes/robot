"""
Affinity matrix construction for spectral decomposition.

Combines semantic feature affinities with color-based affinities
following the paper's methodology.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


def build_feature_affinity(features: torch.Tensor) -> torch.Tensor:
    """
    Build feature affinity matrix from patch features.

    Implements Equation 2 from the paper:
    W_feat = ff^T ⊙ (ff^T > 0)

    Args:
        features: [H, W, D] feature tensor

    Returns:
        W_feat: [H*W, H*W] affinity matrix
    """
    H, W, D = features.shape
    N = H * W

    # Reshape to [N, D] and normalize
    features_flat = features.reshape(N, D)
    features_norm = F.normalize(features_flat, dim=-1)

    # Compute correlations
    affinity = features_norm @ features_norm.T  # [N, N]

    # Threshold negative correlations to zero (as in paper)
    affinity = affinity * (affinity > 0).float()

    return affinity


def build_color_affinity_knn(
    image: torch.Tensor,
    target_size: tuple[int, int],
    k: int = 10
) -> torch.Tensor:
    """
    Build sparse KNN-based color affinity matrix.

    Implements the KNN matting approach from Equation 1 (paper supplement).
    Converts image to HSV and builds k-nearest neighbor graph.

    Args:
        image: [C, H_orig, W_orig] RGB image tensor (values in [0, 1])
        target_size: (H, W) target resolution for downsampling
        k: number of nearest neighbors

    Returns:
        W_knn: [H*W, H*W] sparse affinity matrix
    """
    from skimage import color
    import scipy.sparse as sp
    from sklearn.neighbors import NearestNeighbors

    H, W = target_size
    N = H * W

    # Downsample image to target size
    image_resized = F.interpolate(
        image.unsqueeze(0),
        size=(H, W),
        mode='bilinear',
        align_corners=False
    )[0]  # [C, H, W]

    # Convert to HSV color space
    # Move to numpy for skimage
    image_np = image_resized.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    image_hsv = color.rgb2hsv(image_np)  # [H, W, 3]

    # Create feature vector: [cos(H), sin(H), S, V, y, x]
    # Normalize spatial coordinates to [0, 1]
    y_coords, x_coords = np.meshgrid(np.arange(H) / H, np.arange(W) / W, indexing='ij')

    # Circular encoding of hue
    hue_cos = np.cos(2 * np.pi * image_hsv[:, :, 0])
    hue_sin = np.sin(2 * np.pi * image_hsv[:, :, 0])

    # Stack features: [H, W, 6]
    features = np.stack([
        hue_cos,
        hue_sin,
        image_hsv[:, :, 1],  # Saturation
        image_hsv[:, :, 2],  # Value
        y_coords,
        x_coords
    ], axis=-1)

    features_flat = features.reshape(N, 6)  # [N, 6]

    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(features_flat)
    distances, indices = nbrs.kneighbors(features_flat)

    # Build sparse affinity matrix
    # W(u,v) = 1 - ||ψ(u) - ψ(v)|| if v in KNN(u), else 0
    row_indices = []
    col_indices = []
    values = []

    for i in range(N):
        for j_idx in range(1, k+1):  # Skip self (index 0)
            j = indices[i, j_idx]
            dist = distances[i, j_idx]
            affinity = max(0.0, 1.0 - dist)  # 1 - distance

            if affinity > 0:
                row_indices.append(i)
                col_indices.append(j)
                values.append(affinity)

    # Create sparse matrix and convert to dense
    W_sparse = sp.coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(N, N)
    )

    # Make symmetric
    W_sparse = W_sparse + W_sparse.T

    # Convert to dense torch tensor
    W_knn = torch.from_numpy(W_sparse.toarray()).float()

    return W_knn


def build_affinity_matrix(
    features: torch.Tensor,
    image: Optional[torch.Tensor] = None,
    lambda_knn: float = 10.0,
    knn_neighbors: int = 10,
    intermediate_resolution: int = 64
) -> torch.Tensor:
    """
    Build combined affinity matrix from features and color.

    Implements Equation 3 from the paper:
    W = W_feat + λ_knn * W_knn

    Args:
        features: [H, W, D] feature tensor
        image: [3, H_orig, W_orig] RGB image (optional, for color affinity)
        lambda_knn: weight for color affinity (0 = no color)
        knn_neighbors: number of nearest neighbors for color affinity
        intermediate_resolution: target resolution for color affinity

    Returns:
        W: [H*W, H*W] combined affinity matrix
    """
    H, W, D = features.shape

    # Build feature affinity
    W_feat = build_feature_affinity(features)

    # Optionally add color affinity
    if lambda_knn > 0 and image is not None:
        # Downsample to intermediate resolution
        target_H = min(H, intermediate_resolution)
        target_W = min(W, intermediate_resolution)

        # If features need to be resized to match color resolution
        if H != target_H or W != target_W:
            features_resized = F.interpolate(
                features.permute(2, 0, 1).unsqueeze(0),  # [1, D, H, W]
                size=(target_H, target_W),
                mode='bilinear',
                align_corners=False
            )[0].permute(1, 2, 0)  # [target_H, target_W, D]

            W_feat = build_feature_affinity(features_resized)

        # Build color affinity at same resolution
        W_knn = build_color_affinity_knn(
            image,
            target_size=(target_H, target_W),
            k=knn_neighbors
        ).to(W_feat.device)

        # Combine affinities
        W = W_feat + lambda_knn * W_knn
    else:
        W = W_feat

    return W
