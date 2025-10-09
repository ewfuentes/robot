"""
Spectral methods for unsupervised object detection and segmentation.

Based on "Deep Spectral Methods: A Surprisingly Strong Baseline for
Unsupervised Semantic Segmentation and Localization" (Melas-Kyriazi et al., 2022)
"""

from experimental.overhead_matching.swag.model.spectral.affinity_matrix import build_affinity_matrix
from experimental.overhead_matching.swag.model.spectral.spectral_decomposer import SpectralDecomposer
from experimental.overhead_matching.swag.model.spectral.object_detector import detect_objects_from_eigenvectors
from experimental.overhead_matching.swag.model.spectral.dino_feature_extractor import DinoFeatureHookExtractor

__all__ = [
    'build_affinity_matrix',
    'SpectralDecomposer',
    'detect_objects_from_eigenvectors',
    'DinoFeatureHookExtractor',
]
