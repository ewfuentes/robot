import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from experimental.overhead_matching.swag.model import (
        swag_patch_embedding as spe
    )
    from experimental.overhead_matching.swag.data import (
        vigor_dataset as vd
    )
    from experimental.overhead_matching.swag.model.swag_config_types import (
        SpectralLandmarkExtractorConfig
    )
    from experimental.overhead_matching.swag.model.spectral_landmark_extractor import (
        SpectralLandmarkExtractor
    )

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib
    matplotlib.style.use('ggplot')
    from pathlib import Path
    import torch
    import numpy as np
    import torchvision as tv

    return (
        Path,
        SpectralLandmarkExtractor,
        SpectralLandmarkExtractorConfig,
        mo,
        np,
        patches,
        plt,
        spe,
        torch,
        tv,
        vd,
    )


@app.cell
def _(Path, vd):
    _dataset_path = Path('/data/overhead_matching/datasets/VIGOR/Chicago')

    dataset = vd.VigorDataset(
        _dataset_path,
        config=vd.VigorDatasetConfig(
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None,
        )
    )
    return (dataset,)


@app.cell
def _(SpectralLandmarkExtractor, SpectralLandmarkExtractorConfig):
    # Create extractor with default config
    _config = SpectralLandmarkExtractorConfig(
        dino_model="dinov3_vitb16",
        feature_source="attention_keys",
        lambda_knn=0.0,
        max_landmarks_per_image=10,
        min_bbox_size=20,
        aggregation_method="weighted_mean",
        output_feature_dim=768
    )

    _extractor = SpectralLandmarkExtractor(_config).cuda().eval()
    return


@app.cell
def _(dataset, mo):
    # Get a subset of satellite images for visualization
    sat_dataset = dataset.get_sat_patch_view()

    # Create slider to select which image to visualize
    image_slider = mo.ui.slider(
        start=0,
        stop=min(100, len(sat_dataset)-1),
        value=0,
        label='Image Index',
        show_value=True
    )

    return image_slider, sat_dataset


@app.cell
def _(mo):
    # Create sliders for configuration parameters
    lambda_knn_slider = mo.ui.slider(
        start=0.0,
        stop=20.0,
        step=0.5,
        value=10.0,
        label='Lambda KNN (color weight)',
        show_value=True
    )

    max_landmarks_slider = mo.ui.slider(
        start=1,
        stop=20,
        value=10,
        label='Max Landmarks',
        show_value=True
    )

    feature_source_dropdown = mo.ui.dropdown(
        options=['attention_keys', 'attention_input', 'model_output'],
        value='attention_keys',
        label='Feature Source'
    )

    return feature_source_dropdown, lambda_knn_slider, max_landmarks_slider


@app.cell
def _(
    SpectralLandmarkExtractor,
    SpectralLandmarkExtractorConfig,
    feature_source_dropdown,
    lambda_knn_slider,
    max_landmarks_slider,
):
    # Create extractor with interactive config
    _interactive_config = SpectralLandmarkExtractorConfig(
        dino_model="dinov3_vitb16",
        feature_source=feature_source_dropdown.value,
        lambda_knn=lambda_knn_slider.value,
        max_landmarks_per_image=max_landmarks_slider.value,
        min_bbox_size=20,
        aggregation_method="weighted_mean",
        output_feature_dim=768
    )

    interactive_extractor = SpectralLandmarkExtractor(_interactive_config).cuda().eval()
    return (interactive_extractor,)


@app.cell
def _(
    Path,
    image_slider,
    interactive_extractor,
    sat_dataset,
    spe,
    torch,
    tv,
    vd,
):
    # Get the selected image
    sample = sat_dataset[image_slider.value]
    _image = sample.satellite.unsqueeze(0).cuda()
    _image = tv.io.decode_image(Path('/home/erick/Downloads/2008_000129.jpg'))
    _image = tv.transforms.v2.functional.to_dtype(_image, scale=True)
    sample = vd.VigorDatasetItem(
        satellite=_image,
        satellite_metadata=sample.satellite_metadata,
        panorama_metadata=None,
        panorama=None,
        cached_panorama_tensors=None,
        cached_satellite_tensors=None
    )
    _image = _image.unsqueeze(0).cuda()

    print(_image.shape)

    # Run extractor
    _model_input = spe.ModelInput(
        image=_image,
        metadata=[sample.satellite_metadata]
    )

    with torch.no_grad():
        _output = interactive_extractor(_model_input)

    # Move outputs to CPU for visualization
    features = _output.features.cpu()
    _positions = _output.positions.cpu()
    mask = _output.mask.cpu()
    debug = _output.debug

    # Get number of detected landmarks
    num_detected = (~mask[0]).sum().item()

    return debug, features, mask, num_detected, sample


@app.cell
def _(
    debug,
    image_slider,
    lambda_knn_slider,
    max_landmarks_slider,
    mo,
    num_detected,
):
    # Display configuration and results
    mo.md(f"""
    ## Spectral Landmark Extractor Visualization

    **Configuration:**
    - Image Index: {image_slider.value}
    - Lambda KNN: {lambda_knn_slider.value}
    - Max Landmarks: {max_landmarks_slider.value}

    **Results:**
    - Detected Landmarks: {num_detected}
    - Eigenvalues computed: {len(debug['eigenvalues'][0])}
    """)
    return


@app.cell
def _(debug, mask, mo, np, patches, plt, sample):
    # Visualize detected landmarks with bounding boxes
    _fig, _axes = plt.subplots(1, 2, figsize=(16, 8))

    # Original image
    _ax = _axes[0]
    _ax.imshow(sample.satellite.permute(1, 2, 0).cpu())
    _ax.set_title('Original Image')
    _ax.axis('off')

    # Image with bounding boxes
    _ax = _axes[1]
    _ax.imshow(sample.satellite.permute(1, 2, 0).cpu())

    # Draw bounding boxes for detected landmarks
    _bboxes = debug['bboxes'][0].cpu().numpy()
    _colors = plt.cm.rainbow(np.linspace(0, 1, len(_bboxes)))

    for _i, (_bbox, _color) in enumerate(zip(_bboxes, _colors)):
        if not mask[0, _i]:  # Only draw non-padded landmarks
            _x1, _y1, _x2, _y2 = _bbox
            _width = _x2 - _x1
            _height = _y2 - _y1
            _rect = patches.Rectangle(
                (_x1, _y1), _width, _height,
                linewidth=2, edgecolor=_color, facecolor='none'
            )
            _ax.add_patch(_rect)
            _ax.text(_x1, _y1-5, f'L{_i}', color=_color, fontsize=12,
                   bbox=dict(facecolor='white', alpha=0.7))

    _ax.set_title(f'Detected Landmarks (n={(~mask[0]).sum().item()})')
    _ax.axis('off')

    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    # Create slider to select which eigenvector to visualize
    eigenvector_slider = mo.ui.slider(
        start=0,
        stop=9,
        value=0,
        label='Eigenvector Index',
        show_value=True
    )
    return (eigenvector_slider,)


@app.cell
def _(debug, eigenvector_slider, mo, np, plt, sample):
    # Visualize eigenvectors as heatmaps
    _eigenvectors = debug['eigenvectors'][0].cpu()  # [K, H, W]
    eigenvalues = debug['eigenvalues'][0].cpu().numpy()

    _idx = eigenvector_slider.value

    _fig, _axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    _axes[0].imshow(sample.satellite.permute(1, 2, 0).cpu())
    _axes[0].set_title('Original Image')
    _axes[0].axis('off')

    # Eigenvector heatmap
    _eigvec = _eigenvectors[_idx].numpy()
    _im = _axes[1].imshow(_eigvec, cmap='viridis', interpolation='bilinear')
    _axes[1].set_title(f'Eigenvector {_idx}\n(λ={eigenvalues[_idx]:.4f})')
    plt.colorbar(_im, ax=_axes[1])
    _axes[1].axis('off')

    # Thresholded eigenvector (shows segmentation)
    _threshold = np.percentile(_eigvec, 75)
    _binary_mask = _eigvec > _threshold
    _axes[2].imshow(sample.satellite.permute(1, 2, 0).cpu())
    _axes[2].imshow(_binary_mask, alpha=0.5, cmap='Reds', interpolation='nearest')
    _axes[2].set_title(f'Thresholded (75th percentile)')
    _axes[2].axis('off')

    plt.tight_layout()

    mo.vstack([
        mo.md(f"### Eigenvector Visualization"),
        eigenvector_slider,
        mo.mpl.interactive(_fig)
    ])
    return (eigenvalues,)


@app.cell
def _(debug, mo, np, plt, sample):
    # Visualize object masks (soft segmentation masks)
    _object_masks = debug['object_masks'][0].cpu()  # [N_detected, H, W]
    _num_objects = _object_masks.shape[0]


    # Create grid visualization
    _ncols = min(4, _num_objects)
    _nrows = (_num_objects + _ncols - 1) // _ncols

    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(4*_ncols, 4*_nrows))
    if _nrows == 1 and _ncols == 1:
        _axes = np.array([[_axes]])
    elif _nrows == 1 or _ncols == 1:
        _axes = _axes.reshape(_nrows, _ncols)

    for _i in range(_num_objects):
        _row = _i // _ncols
        _col = _i % _ncols
        _ax = _axes[_row, _col]

        # Show original image with mask overlay
        _ax.imshow(sample.satellite.permute(1, 2, 0).cpu())
        _mask_overlay = _object_masks[_i].numpy()
        _ax.imshow(_mask_overlay, alpha=0.6, cmap='hot', interpolation='bilinear')
        _ax.set_title(f'Object {_i}')
        _ax.axis('off')

    # Hide empty subplots
    for _i in range(_num_objects, _nrows * _ncols):
        _row = _i // _ncols
        _col = _i % _ncols
        _axes[_row, _col].axis('off')

    plt.tight_layout()
    mo.vstack([
        mo.md(f"### Object Segmentation Masks (n={_num_objects})"),
        mo.mpl.interactive(_fig)])

    return


@app.cell
def _(eigenvalues, mo, plt):
    # Plot eigenvalue spectrum
    _fig, _ax = plt.subplots(1, 1, figsize=(10, 5))

    _ax.plot(eigenvalues, 'o-', linewidth=2, markersize=8)
    _ax.set_xlabel('Eigenvector Index')
    _ax.set_ylabel('Eigenvalue')
    _ax.set_title('Eigenvalue Spectrum\n(smaller eigenvalues = stronger object signal)')
    _ax.grid(True, alpha=0.3)

    plt.tight_layout()

    mo.md("### Eigenvalue Spectrum") + mo.mpl.interactive(_fig)
    return


@app.cell
def _(features, mask, mo, plt, torch):
    # Visualize feature similarity between detected landmarks
    _valid_features = features[0][~mask[0]]  # [N_detected, D]

    if _valid_features.shape[0] > 1:
        # Compute pairwise cosine similarity
        _normalized_features = torch.nn.functional.normalize(_valid_features, dim=1)
        _similarity_matrix = (_normalized_features @ _normalized_features.T).numpy()

        _fig, _ax = plt.subplots(1, 1, figsize=(8, 8))
        _im = _ax.imshow(_similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        _ax.set_title('Landmark Feature Similarity (Cosine)')
        _ax.set_xlabel('Landmark Index')
        _ax.set_ylabel('Landmark Index')
        plt.colorbar(_im, ax=_ax)

        # Add text annotations
        for _i in range(_similarity_matrix.shape[0]):
            for _j in range(_similarity_matrix.shape[1]):
                _text = _ax.text(_j, _i, f'{_similarity_matrix[_i, _j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

        plt.tight_layout()
        mo.md("### Landmark Feature Similarity") + mo.mpl.interactive(_fig)
    else:
        mo.md("### Need at least 2 landmarks to compute similarity")
    return


@app.cell
def _(debug, mask, mo, np, plt):
    # Summary statistics
    _eigenvector_indices = debug['eigenvector_indices'][0][~mask[0]].cpu().numpy()
    _landmark_eigenvalues = debug['eigenvalues'][0][~mask[0]].cpu().numpy()

    if len(_landmark_eigenvalues) > 0:
        _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram of which eigenvectors were used
        _axes[0].hist(_eigenvector_indices, bins=np.arange(0, 11)-0.5, edgecolor='black')
        _axes[0].set_xlabel('Eigenvector Index')
        _axes[0].set_ylabel('Number of Landmarks')
        _axes[0].set_title('Distribution of Source Eigenvectors')
        _axes[0].grid(True, alpha=0.3)

        # Eigenvalues of detected landmarks
        _axes[1].bar(range(len(_landmark_eigenvalues)), _landmark_eigenvalues)
        _axes[1].set_xlabel('Landmark Index')
        _axes[1].set_ylabel('Eigenvalue')
        _axes[1].set_title('Eigenvalues of Detected Landmarks')
        _axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        mo.md("### Landmark Statistics") + mo.mpl.interactive(_fig)
    else:
        mo.md("### No landmarks detected")
    return


@app.cell
def _(mo):
    # Display controls
    mo.md("""
    ## Interactive Controls

    Adjust the parameters below to see how they affect landmark detection:
    """)
    return


@app.cell
def _(
    feature_source_dropdown,
    image_slider,
    lambda_knn_slider,
    max_landmarks_slider,
    mo,
):
    mo.vstack([
        image_slider,
        lambda_knn_slider,
        max_landmarks_slider,
        feature_source_dropdown
    ])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
