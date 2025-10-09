"""
Object detection from eigenvectors of the Laplacian.

Converts soft eigensegments into discrete bounding boxes and features.
"""

import torch
import numpy as np
from skimage import measure
from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectedObject:
    """Represents a detected object from spectral decomposition."""
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in absolute pixels
    feature: torch.Tensor  # [D] exemplar feature vector
    eigenvector_idx: int  # Which eigenvector detected this object
    eigenvalue: float  # Corresponding eigenvalue
    mask: torch.Tensor  # [H, W] binary mask
    soft_mask: torch.Tensor  # [H, W] soft mask from eigenvector


def detect_objects_from_eigenvectors(
    eigenvectors: torch.Tensor,
    eigenvalues: torch.Tensor,
    features: torch.Tensor,
    min_bbox_size: int = 20,
    max_objects: int = 10,
    aggregation_method: str = "weighted_mean"
) -> list[DetectedObject]:
    """
    Detect objects from eigenvectors using spectral bisection.

    Args:
        eigenvectors: [H*W, K] eigenvectors (columns)
        eigenvalues: [K] corresponding eigenvalues
        features: [H, W, D] feature tensor
        min_bbox_size: Minimum bounding box size (width or height)
        max_objects: Maximum number of objects to return
        aggregation_method: How to compute exemplar feature ("weighted_mean", "mean", "max")

    Returns:
        List of detected objects (up to max_objects)
    """
    H, W, D = features.shape
    N = H * W
    K = eigenvectors.shape[1]

    detected_objects = []

    # Process each eigenvector
    for k in range(min(K, max_objects)):
        eigenvector = eigenvectors[:, k].reshape(H, W)  # [H, W]
        eigenvalue = eigenvalues[k].item()

        # Threshold at 0 (spectral bisection)
        binary_mask = (eigenvector > 0).cpu().numpy()

        # Find connected components
        labeled_mask = measure.label(binary_mask, connectivity=2)

        if labeled_mask.max() == 0:
            # No objects found in this eigenvector
            continue

        # Get the largest connected component
        region_sizes = np.bincount(labeled_mask.flat)[1:]  # Exclude background (0)
        if len(region_sizes) == 0:
            continue

        largest_region_label = region_sizes.argmax() + 1
        object_mask = (labeled_mask == largest_region_label)

        # Get bounding box
        rows, cols = np.where(object_mask)
        if len(rows) == 0:
            continue

        y1, y2 = rows.min(), rows.max() + 1
        x1, x2 = cols.min(), cols.max() + 1

        # Filter by minimum size
        if (x2 - x1) < min_bbox_size or (y2 - y1) < min_bbox_size:
            continue

        # Convert mask to torch
        object_mask_torch = torch.from_numpy(object_mask).to(features.device)

        # Compute exemplar feature
        feature = compute_exemplar_feature(
            features,
            object_mask_torch,
            eigenvector,
            method=aggregation_method
        )

        # Store detected object
        detected_objects.append(DetectedObject(
            bbox=(x1, y1, x2, y2),
            feature=feature,
            eigenvector_idx=k,
            eigenvalue=eigenvalue,
            mask=object_mask_torch,
            soft_mask=eigenvector.abs()  # Use absolute value for soft mask
        ))

        # Stop if we have enough objects
        if len(detected_objects) >= max_objects:
            break

    return detected_objects


def compute_exemplar_feature(
    features: torch.Tensor,
    mask: torch.Tensor,
    eigenvector: torch.Tensor,
    method: str = "weighted_mean"
) -> torch.Tensor:
    """
    Compute an exemplar feature vector for a detected object.

    Args:
        features: [H, W, D] feature tensor
        mask: [H, W] binary mask
        eigenvector: [H, W] eigenvector values (for weighting)
        method: "weighted_mean", "mean", or "max"

    Returns:
        feature: [D] exemplar feature vector
    """
    H, W, D = features.shape

    if method == "mean":
        # Simple average of features in the mask
        masked_features = features[mask]  # [N_masked, D]
        if len(masked_features) == 0:
            return torch.zeros(D, device=features.device)
        feature = masked_features.mean(dim=0)

    elif method == "weighted_mean":
        # Weighted average using eigenvector values as attention
        weights = eigenvector[mask].abs()  # [N_masked]
        masked_features = features[mask]  # [N_masked, D]

        if len(masked_features) == 0 or weights.sum() < 1e-12:
            return torch.zeros(D, device=features.device)

        # Normalize weights
        weights = weights / weights.sum()

        # Weighted sum
        feature = (masked_features * weights.unsqueeze(-1)).sum(dim=0)

    elif method == "max":
        # Max pooling across each feature dimension
        masked_features = features[mask]  # [N_masked, D]
        if len(masked_features) == 0:
            return torch.zeros(D, device=features.device)
        feature = masked_features.max(dim=0)[0]

    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    return feature
