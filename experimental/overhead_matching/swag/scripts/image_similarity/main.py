#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from experimental.overhead_matching.swag.scripts.image_similarity.interactive_viewer import InteractiveImageViewer
from experimental.overhead_matching.swag.scripts.image_similarity.enhanced_interactive_viewer import EnhancedInteractiveViewer
from experimental.overhead_matching.swag.scripts.image_similarity.vigor_browser import VigorBrowser
from experimental.overhead_matching.swag.data.vigor_dataset import VigorDatasetConfig, SampleMode


def validate_image_paths(image_paths: List[str]) -> List[str]:
    """Validate that all image paths exist and are valid image files."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    validated_paths = []

    for path_str in image_paths:
        path = Path(path_str)

        if not path.exists():
            print(f"Error: Image file does not exist: {path}")
            sys.exit(1)

        if path.suffix.lower() not in valid_extensions:
            print(f"Error: Unsupported image format: {path}")
            print(f"Supported formats: {', '.join(valid_extensions)}")
            sys.exit(1)

        validated_paths.append(str(path.resolve()))

    return validated_paths


def main():
    parser = argparse.ArgumentParser(
        description="Interactive image similarity viewer using DINOv3 features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # View similarity between two images
  python -m experimental.overhead_matching.swag.scripts.image_similarity.main \\
    image1.jpg image2.jpg

  # View similarity across multiple images
  python -m experimental.overhead_matching.swag.scripts.image_similarity.main \\
    img1.jpg img2.jpg img3.jpg img4.jpg

  # Use custom figure size
  python -m experimental.overhead_matching.swag.scripts.image_similarity.main \\
    --figsize 20 10 image1.jpg image2.jpg

Instructions:
  - Click on any point in any image to see similarity heatmaps
  - Red cross (+) marks the clicked location
  - Hot colors (red/yellow) indicate high similarity
  - Cool colors (blue/black) indicate low similarity
  - Use 'Clear' button to remove overlays
  - Close window or Ctrl+C to exit
        """
    )

    parser.add_argument(
        'images',
        nargs='*',
        help='Paths to image files to compare (2 or more images, not needed in VIGOR mode)'
    )

    parser.add_argument(
        '--figsize',
        nargs=2,
        type=int,
        default=[16, 8],
        metavar=('WIDTH', 'HEIGHT'),
        help='Figure size in inches (default: 16 8)'
    )

    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use for computation (default: auto)'
    )

    parser.add_argument(
        '--model',
        default='dinov2_vitb14',
        choices=[
            'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14',
            'dinov3_vits16', 'dinov3_vits16plus', 'dinov3_vitb16', 'dinov3_vitl16',
            'dinov3_vith16plus', 'dinov3_vit7b16', 'dinov3_convnext_tiny', 'dinov3_convnext_small'
        ],
        help='DINO model to use (default: dinov2_vitb14)'
    )

    # VIGOR dataset options
    parser.add_argument(
        '--vigor-dataset-path',
        type=str,
        help='Path to VIGOR dataset directory (enables VIGOR browse mode)'
    )

    # PCA analysis options
    parser.add_argument(
        '--pca-mode',
        action='store_true',
        help='Enable PCA visualization mode'
    )

    parser.add_argument(
        '--num-components',
        type=int,
        default=10,
        help='Number of PCA components to compute (default: 10)'
    )

    # Landmark overlay options
    parser.add_argument(
        '--landmarks',
        type=str,
        help='Path to landmarks GeoJSON file (e.g., v3.geojson)'
    )


    args = parser.parse_args()

    # Validate mode selection
    vigor_mode = args.vigor_dataset_path is not None
    file_mode = len(args.images) > 0

    if vigor_mode and file_mode:
        print("Error: Cannot use both VIGOR dataset and file paths simultaneously")
        sys.exit(1)

    if not vigor_mode and not file_mode:
        print("Error: Must provide either image files or --vigor-dataset-path")
        sys.exit(1)

    # Validate file mode inputs
    if file_mode:
        if len(args.images) < 2:
            print("Error: Please provide at least 2 images to compare")
            sys.exit(1)

        if len(args.images) > 6:
            print("Warning: Displaying more than 6 images may be difficult to view")

        validated_paths = validate_image_paths(args.images)

    # Validate VIGOR mode inputs
    if vigor_mode:
        vigor_path = Path(args.vigor_dataset_path)
        if not vigor_path.exists():
            print(f"Error: VIGOR dataset path does not exist: {vigor_path}")
            sys.exit(1)

    # Validate landmarks file
    if args.landmarks:
        landmarks_path = Path(args.landmarks)
        if not landmarks_path.exists():
            print(f"Error: Landmarks file does not exist: {landmarks_path}")
            sys.exit(1)

    def launch_similarity_viewer(image_data: Dict[str, Any]):
        """Launch the similarity viewer with given image data."""
        print(f"\nUsing model: {args.model}")
        print(f"Device: {args.device}")

        # Determine if we need enhanced features
        use_enhanced = args.pca_mode or args.landmarks

        if use_enhanced:
            print("Starting enhanced interactive viewer...")
            if args.pca_mode:
                print("- PCA analysis enabled")
            if args.landmarks:
                print(f"- Landmark overlays enabled: {args.landmarks}")

            device = 'cuda' if args.device == 'auto' else args.device
            viewer = EnhancedInteractiveViewer(
                image_data=image_data,
                figsize=tuple(args.figsize),
                pca_enabled=args.pca_mode,
                landmarks_path=args.landmarks,
                model_str=args.model,
                device=device
            )
        else:
            print("Starting basic interactive viewer...")
            print("Click on any point in any image to see similarity heatmaps!")

            # Convert image_data to list of paths for basic viewer
            image_paths = []
            for img_id, img_input in image_data.items():
                if isinstance(img_input, str):
                    image_paths.append(img_input)
                else:
                    print("Warning: Basic viewer only supports file paths. Use enhanced viewer for VIGOR data.")
                    return

            viewer = InteractiveImageViewer(
                image_paths=image_paths,
                figsize=tuple(args.figsize)
            )

            # Override the similarity engine model if specified
            if args.model != 'dinov2_vitb14':
                from experimental.overhead_matching.swag.scripts.image_similarity.similarity_engine import ImageSimilarityEngine
                device = 'cuda' if args.device == 'auto' else args.device
                viewer.similarity_engine = ImageSimilarityEngine(
                    model_str=args.model,
                    device=device
                )

        viewer.show()

    if vigor_mode:
        # VIGOR browser mode
        print(f"Opening VIGOR dataset browser: {args.vigor_dataset_path}")

        config = VigorDatasetConfig(
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None,
            sample_mode=SampleMode.NEAREST,
            should_load_images=True
        )

        browser = VigorBrowser(args.vigor_dataset_path, config)
        image_data = browser.show()

        # If user selected images, launch the viewer
        if image_data:
            launch_similarity_viewer(image_data)
        else:
            print("No images selected. Exiting...")

    else:
        # File mode
        print(f"Loading {len(validated_paths)} images:")
        for i, path in enumerate(validated_paths, 1):
            print(f"  {i}. {Path(path).name}")

        # Convert file paths to image_data format
        image_data = {f"image_{i}": path for i, path in enumerate(validated_paths)}
        launch_similarity_viewer(image_data)


if __name__ == "__main__":
    main()