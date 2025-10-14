# Spectral Landmark Extractor

Implementation of "Deep Spectral Methods: A Surprisingly Strong Baseline for Unsupervised Semantic Segmentation and Localization" (Melas-Kyriazi et al., 2022) integrated into the SWAG pipeline.

## Overview

This extractor uses spectral decomposition of DINOv3 features to detect objects/landmarks in images without supervision. It extracts attention features from a DINOv3 vision transformer, builds a graph Laplacian from semantic and color affinities, and uses eigenvectors to identify object regions.

## Components

- `affinity_matrix.py` - Builds semantic + color affinity matrices
- `spectral_decomposer.py` - Eigendecomposition of graph Laplacian
- `object_detector.py` - Converts eigenvectors → bounding boxes + features
- `dino_feature_extractor.py` - Extracts DINOv3 features with forward hooks

## Configuration

```yaml
extractor_config_by_name:
  spectral_landmarks:
    dino_model: dinov3_vitb16        # DINOv3 ViT model
    feature_source: attention_keys    # attention_keys, attention_input, model_output
    lambda_knn: 10.0                 # Color affinity weight (0 = no color)
    max_landmarks_per_image: 10      # Max objects to detect
    min_bbox_size: 20                # Min bbox size in pixels
    aggregation_method: weighted_mean # Feature pooling method
    output_feature_dim: 768          # Output feature dimension
```

## Bazel Build

```bash
# Build the extractor
bazel build //experimental/overhead_matching/swag/model:spectral_landmark_extractor

# Run the tests
bazel test //experimental/overhead_matching/swag/scripts:spectral_landmark_extractor_test

# Run the visualization notebook
marimo edit experimental/overhead_matching/swag/scripts/spectral_landmarks_visualization.py
```

## Caching

Works with the existing tensor caching infrastructure:

```bash
python experimental/overhead_matching/swag/scripts/populate_tensor_cache.py \
  --train_config configs/your_config.yaml \
  --field_spec "sat_model_config.extractor_config_by_name.spectral_landmarks" \
  --dataset /path/to/dataset
```

## Output Format

Returns `ExtractorOutput` with:
- `features`: [batch, max_landmarks, output_dim] - Projected landmark features
- `positions`: [batch, max_landmarks, 2] - Bbox centers relative to image center
- `mask`: [batch, max_landmarks] - True = padding
- `debug`: Dictionary with bboxes, eigenvectors, eigenvalues, masks

## Dependencies

All required dependencies are already in `third_party/python/requirements_3_12.txt`:
- scipy
- scikit-learn
- scikit-image
- torchvision (required by DINOv3 torch.hub loading)

## Paper Reference

Melas-Kyriazi, L., Rupprecht, C., Laina, I., & Vedaldi, A. (2022). Deep Spectral Methods: A Surprisingly Strong Baseline for Unsupervised Semantic Segmentation and Localization. arXiv:2205.07839.
