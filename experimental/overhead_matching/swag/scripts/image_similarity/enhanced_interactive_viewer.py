import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, RadioButtons, Slider, CheckButtons
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional, Dict, Union, Any
from enum import Enum

from experimental.overhead_matching.swag.scripts.image_similarity.similarity_engine import ImageSimilarityEngine
from experimental.overhead_matching.swag.scripts.image_similarity.pca_analyzer import PCAAnalyzer
from experimental.overhead_matching.swag.scripts.image_similarity.landmark_overlay import LandmarkOverlay


class ViewMode(Enum):
    SIMILARITY = "similarity"
    PCA = "pca"
    LANDMARKS = "landmarks"


class EnhancedInteractiveViewer:
    def __init__(self, image_data: Dict[str, Union[str, Dict[str, Any]]],
                 figsize: tuple = (18, 10),
                 pca_enabled: bool = False,
                 landmarks_path: Optional[str] = None,
                 model_str: str = "dinov2_vitb14",
                 device: str = "cuda"):
        """
        Enhanced interactive image viewer with multi-mode support.

        Args:
            image_data: Dict mapping image_id to either file path or VIGOR data dict
            figsize: Figure size
            pca_enabled: Whether to enable PCA mode
            landmarks_path: Path to landmarks GeoJSON file
            model_str: DINO model to use for feature extraction
            device: Device to use for computation ('cuda' or 'cpu')
        """
        self.image_data = image_data
        self.image_ids = list(image_data.keys())
        self.num_images = len(self.image_ids)

        # Initialize engines
        self.similarity_engine = ImageSimilarityEngine(model_str=model_str, device=device)
        self.pca_analyzer = PCAAnalyzer() if pca_enabled else None
        self.landmark_overlay = LandmarkOverlay(landmarks_path) if landmarks_path else None

        # Current mode
        self.current_mode = ViewMode.SIMILARITY
        self.pca_enabled = pca_enabled
        self.landmarks_enabled = landmarks_path is not None

        # PCA state (must be initialized before _load_images)
        self.pca_fitted = False
        self.current_pca_component = 0
        self.max_pca_components = 10
        self.pca_rgb_mode = True  # True = RGB mode, False = single component mode
        self.pca_r_component = 0
        self.pca_g_component = 1
        self.pca_b_component = 2

        # Interaction state
        self.clicked_point = None
        self.current_similarities = {}
        self.current_pca_heatmaps = {}

        # Load and preprocess all images
        self.images = {}  # image_id -> numpy array
        self.image_metadata = {}  # image_id -> metadata dict
        self._load_images()

        # UI components
        self.fig = None
        self.axes = []
        self.mode_selector = None
        self.pca_slider = None
        self.pca_r_slider = None
        self.pca_g_slider = None
        self.pca_b_slider = None
        self.pca_mode_checkbox = None
        self.buttons = {}
        self.landmark_checkboxes = None

        # Marquee title
        self.title_text = None
        self.title_offset = 0
        self.marquee_animation = None

        self._setup_ui(figsize)

    def _load_images(self):
        """Load all images and extract features."""
        print(f"Loading {self.num_images} images and extracting features...")

        for image_id, image_input in self.image_data.items():
            if isinstance(image_input, str):
                # File path
                img = Image.open(image_input).convert('RGB')
                self.images[image_id] = np.array(img)
                self.image_metadata[image_id] = {'source': 'file', 'path': image_input}
            else:
                # VIGOR data
                self.images[image_id] = (image_input['array'] * 255).astype(np.uint8) if image_input['array'].max() <= 1.0 else image_input['array']
                self.image_metadata[image_id] = image_input.get('metadata', {})

            # Extract features for similarity and PCA
            self.similarity_engine.extract_features(image_input)

        # Fit PCA if enabled
        if self.pca_analyzer:
            print("Computing PCA...")
            self.pca_analyzer.fit(
                self.similarity_engine.image_features,
                self.similarity_engine.image_positions,
                self.similarity_engine.image_shapes
            )
            self.pca_fitted = True
            # Heatmaps will be computed lazily on-demand

    def _compute_pca_heatmaps(self):
        """Precompute PCA heatmaps for all images and components."""
        if not self.pca_fitted:
            return

        self.current_pca_heatmaps = {}
        for image_id in self.image_ids:
            self.current_pca_heatmaps[image_id] = {}
            for comp_idx in range(self.max_pca_components):
                try:
                    heatmap = self.pca_analyzer.get_component_heatmap(image_id, comp_idx)
                    self.current_pca_heatmaps[image_id][comp_idx] = heatmap
                except Exception as e:
                    print(f"Error computing PCA heatmap for {image_id}, component {comp_idx}: {e}")

    def _setup_ui(self, figsize):
        """Setup the enhanced UI with multiple modes."""
        self.fig = plt.figure(figsize=figsize)

        # Create marquee title
        self.full_title = "The Absurdly Magnificent Extraordinarily Whimsical Incredibly Fantastical Somewhat Confusing But Definitely Similarity-Related DINO-Vision Wonder-Gadget of Frankly Unnecessary Complexity (Now With Polygons!)"
        self.title_text = self.fig.text(0.5, 0.98, '', fontsize=14, ha='center', va='top',
                                         weight='bold', style='italic')
        self._setup_marquee()

        # Calculate layout for images
        if self.num_images <= 3:
            img_rows, img_cols = 1, self.num_images
        elif self.num_images <= 6:
            img_rows, img_cols = 2, 3
        else:
            img_rows, img_cols = 3, 4

        # Create GridSpec: image rows (each 3 units tall) + 1 control row
        total_rows = img_rows * 3 + 1
        gs = GridSpec(total_rows, 8, figure=self.fig, hspace=0.4, wspace=0.3)

        # Create image display axes
        self.axes = []
        for i in range(self.num_images):
            row = i // img_cols
            col = i % img_cols
            # Each image spans 3 rows in height, 2 columns in width
            col_span = 8 // img_cols  # Distribute columns evenly
            ax = self.fig.add_subplot(gs[row * 3:(row + 1) * 3, col * col_span:(col + 1) * col_span])
            ax.set_xticks([])
            ax.set_yticks([])
            self.axes.append(ax)

        # Create control panel in the last row
        control_row = img_rows * 3
        self._setup_controls(gs, control_row)

        # Initial display
        self._update_display()

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    def _setup_marquee(self):
        """Setup the marquee scrolling title animation."""
        self.display_width = 80  # Number of characters to display at once

        def update_marquee(frame):
            # Create a scrolling window of the title
            extended_title = self.full_title + "    ★    "  # Add separator
            total_len = len(extended_title)

            # Calculate current position
            self.title_offset = (self.title_offset + 1) % total_len

            # Create visible portion (with wrapping)
            if self.title_offset + self.display_width <= total_len:
                visible_text = extended_title[self.title_offset:self.title_offset + self.display_width]
            else:
                # Wrap around
                part1 = extended_title[self.title_offset:]
                part2 = extended_title[:self.display_width - len(part1)]
                visible_text = part1 + part2

            self.title_text.set_text(visible_text)
            return [self.title_text]

        # Create animation with blit=True to avoid redrawing the entire figure (including expensive PCA overlays)
        self.marquee_animation = FuncAnimation(
            self.fig, update_marquee, interval=100, blit=True, cache_frame_data=False
        )

    def _setup_controls(self, gs, control_row):
        """Setup control panel."""
        col_idx = 0

        # Mode selector
        if self.pca_enabled or self.landmarks_enabled:
            modes = ['Similarity']
            if self.pca_enabled:
                modes.append('PCA')
            if self.landmarks_enabled:
                modes.append('Landmarks')

            ax_mode = self.fig.add_subplot(gs[control_row, col_idx])
            self.mode_selector = RadioButtons(ax_mode, modes)
            self.mode_selector.on_clicked(self._change_mode)
            col_idx += 1

        # PCA component sliders
        if self.pca_enabled:
            # RGB mode sliders
            ax_r = self.fig.add_subplot(gs[control_row, col_idx])
            self.pca_r_slider = Slider(ax_r, 'R', 0, self.max_pca_components - 1,
                                      valinit=self.pca_r_component, valfmt='%d', color='red')
            self.pca_r_slider.on_changed(self._change_pca_r)
            col_idx += 1

            ax_g = self.fig.add_subplot(gs[control_row, col_idx])
            self.pca_g_slider = Slider(ax_g, 'G', 0, self.max_pca_components - 1,
                                      valinit=self.pca_g_component, valfmt='%d', color='green')
            self.pca_g_slider.on_changed(self._change_pca_g)
            col_idx += 1

            ax_b = self.fig.add_subplot(gs[control_row, col_idx])
            self.pca_b_slider = Slider(ax_b, 'B', 0, self.max_pca_components - 1,
                                      valinit=self.pca_b_component, valfmt='%d', color='blue')
            self.pca_b_slider.on_changed(self._change_pca_b)
            col_idx += 1

        # Control buttons
        ax_clear = self.fig.add_subplot(gs[control_row, col_idx])
        self.buttons['clear'] = Button(ax_clear, 'Clear')
        self.buttons['clear'].on_clicked(self._clear_overlays)
        col_idx += 1

        if self.landmarks_enabled:
            # Landmark category checkboxes
            ax_landmarks = self.fig.add_subplot(gs[control_row, col_idx:])
            categories = sorted(list(self.landmark_overlay.categories))[:6]  # Limit to 6 for space
            self.landmark_checkboxes = CheckButtons(ax_landmarks, categories,
                                                  [True] * len(categories))
            self.landmark_checkboxes.on_clicked(self._toggle_landmark_category)

    def _update_display(self):
        """Update display based on current mode."""
        # Clear all axes
        for ax in self.axes:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

        # Display images with overlays based on mode
        for i, image_id in enumerate(self.image_ids):
            if i >= len(self.axes):
                break

            ax = self.axes[i]
            image = self.images[image_id]

            # Display base image
            ax.imshow(image)

            # Add mode-specific overlays
            if self.current_mode == ViewMode.SIMILARITY:
                self._add_similarity_overlay(ax, image_id)
            elif self.current_mode == ViewMode.PCA and self.pca_fitted:
                self._add_pca_overlay(ax, image_id)
            elif self.current_mode == ViewMode.LANDMARKS and self.landmarks_enabled:
                self._add_landmark_overlay(ax, image_id, image.shape)

            # Add title
            title = f"{image_id}"
            if image_id in self.image_metadata:
                metadata = self.image_metadata[image_id]
                if 'type' in metadata:
                    title += f" ({metadata['type']})"
                if 'lat' in metadata and 'lon' in metadata:
                    title += f"\n({metadata['lat']:.4f}, {metadata['lon']:.4f})"

            ax.set_title(title, fontsize=10)

            # Add click marker if this image was clicked
            if (self.clicked_point and
                self.clicked_point[0] == image_id and
                self.current_mode == ViewMode.SIMILARITY):
                x, y = self.clicked_point[1], self.clicked_point[2]
                ax.plot(x, y, 'r+', markersize=15, markeredgewidth=3)

        plt.draw()

    def _add_similarity_overlay(self, ax, image_id):
        """Add similarity heatmap overlay."""
        if image_id in self.current_similarities:
            heatmap = self.similarity_engine.similarity_to_heatmap(
                self.current_similarities[image_id], image_id
            )
            ax.imshow(heatmap, alpha=0.6, cmap='hot', vmin=0, vmax=1)

    def _add_pca_overlay(self, ax, image_id):
        """Add PCA component heatmap overlay (computed lazily on-demand)."""
        if not self.pca_fitted:
            return

        # Ensure image_id exists in cache dict
        if image_id not in self.current_pca_heatmaps:
            self.current_pca_heatmaps[image_id] = {}

        if self.pca_rgb_mode:
            # RGB mode: map 3 components to RGB channels
            components_needed = [self.pca_r_component, self.pca_g_component, self.pca_b_component]

            # Compute any missing heatmaps on-demand
            for comp_idx in components_needed:
                if comp_idx not in self.current_pca_heatmaps[image_id]:
                    try:
                        heatmap = self.pca_analyzer.get_component_heatmap(image_id, comp_idx)
                        self.current_pca_heatmaps[image_id][comp_idx] = heatmap
                    except Exception as e:
                        print(f"Error computing PCA heatmap for {image_id}, component {comp_idx}: {e}")
                        return

            # Get cached heatmaps
            r_heatmap = self.current_pca_heatmaps[image_id][self.pca_r_component]
            g_heatmap = self.current_pca_heatmaps[image_id][self.pca_g_component]
            b_heatmap = self.current_pca_heatmaps[image_id][self.pca_b_component]

            # Normalize each component independently to [0, 1]
            r_norm = (r_heatmap - r_heatmap.min()) / (r_heatmap.max() - r_heatmap.min() + 1e-8)
            g_norm = (g_heatmap - g_heatmap.min()) / (g_heatmap.max() - g_heatmap.min() + 1e-8)
            b_norm = (b_heatmap - b_heatmap.min()) / (b_heatmap.max() - b_heatmap.min() + 1e-8)

            # Stack into RGB image
            rgb_image = np.stack([r_norm, g_norm, b_norm], axis=-1)
            ax.imshow(rgb_image, alpha=0.6)
        else:
            # Single component mode - compute on-demand if needed
            if self.current_pca_component not in self.current_pca_heatmaps[image_id]:
                try:
                    heatmap = self.pca_analyzer.get_component_heatmap(image_id, self.current_pca_component)
                    self.current_pca_heatmaps[image_id][self.current_pca_component] = heatmap
                except Exception as e:
                    print(f"Error computing PCA heatmap for {image_id}, component {self.current_pca_component}: {e}")
                    return

            heatmap = self.current_pca_heatmaps[image_id][self.current_pca_component]
            # Normalize heatmap for better visualization
            normalized_heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            ax.imshow(normalized_heatmap, alpha=0.6, cmap='coolwarm', vmin=0, vmax=1)

    def _add_landmark_overlay(self, ax, image_id, image_shape):
        """Add landmark overlay."""
        if not self.landmarks_enabled:
            return

        metadata = self.image_metadata.get(image_id, {})
        if 'lat' in metadata and 'lon' in metadata:
            try:
                # Get visible categories from checkboxes
                visible_categories = set()
                if self.landmark_checkboxes:
                    for i, category in enumerate(self.landmark_checkboxes.labels):
                        if self.landmark_checkboxes.get_status()[i]:
                            visible_categories.add(category.get_text())

                self.landmark_overlay.overlay_on_image(
                    ax, metadata, image_shape[:2], visible_categories, show_labels=True
                )
            except Exception as e:
                print(f"Error adding landmarks to {image_id}: {e}")

    def _on_click(self, event):
        """Handle click events."""
        if event.inaxes not in self.axes:
            return

        # Find which image was clicked
        clicked_image_idx = self.axes.index(event.inaxes)
        if clicked_image_idx >= len(self.image_ids):
            return

        clicked_image_id = self.image_ids[clicked_image_idx]
        clicked_x = int(event.xdata) if event.xdata is not None else 0
        clicked_y = int(event.ydata) if event.ydata is not None else 0

        if self.current_mode == ViewMode.SIMILARITY:
            self._handle_similarity_click(clicked_image_id, clicked_x, clicked_y)
        elif self.current_mode == ViewMode.LANDMARKS and self.landmarks_enabled:
            self._handle_landmark_click(clicked_image_id, clicked_x, clicked_y)

    def _handle_similarity_click(self, image_id, x, y):
        """Handle clicks in similarity mode."""
        print(f"Computing similarities for click at ({x}, {y}) in {image_id}")

        # Store click point
        self.clicked_point = (image_id, x, y)

        # Compute similarities
        try:
            clicked_input = self.image_data[image_id]
            target_inputs = [self.image_data[img_id] for img_id in self.image_ids]

            self.current_similarities = self.similarity_engine.compute_similarities(
                clicked_input, x, y, target_inputs
            )

            # Update display
            self._update_display()

        except Exception as e:
            print(f"Error computing similarities: {e}")

    def _handle_landmark_click(self, image_id, x, y):
        """Handle clicks in landmark mode."""
        if not self.landmarks_enabled:
            return

        metadata = self.image_metadata.get(image_id, {})
        image_shape = self.images[image_id].shape[:2]  # (height, width)
        landmark = self.landmark_overlay.get_landmark_info(x, y, metadata, image_shape)

        if landmark:
            print(f"Clicked landmark: {landmark.name or 'Unnamed'}")
            print(f"Category: {landmark.category}")
            print(f"Properties: {landmark.properties}")
        else:
            print(f"No landmark found at ({x}, {y})")

    def _change_mode(self, label):
        """Change viewing mode."""
        mode_map = {
            'Similarity': ViewMode.SIMILARITY,
            'PCA': ViewMode.PCA,
            'Landmarks': ViewMode.LANDMARKS
        }

        new_mode = mode_map.get(label, ViewMode.SIMILARITY)
        if new_mode != self.current_mode:
            self.current_mode = new_mode
            print(f"Switched to {label} mode")
            self._update_display()

    def _change_pca_component(self, val):
        """Change PCA component."""
        self.current_pca_component = int(val)
        if self.current_mode == ViewMode.PCA:
            print(f"Showing PCA component {self.current_pca_component}")
            self._update_display()

    def _change_pca_r(self, val):
        """Change PCA R component."""
        self.pca_r_component = int(val)
        if self.current_mode == ViewMode.PCA and self.pca_rgb_mode:
            print(f"RGB: R=PC{self.pca_r_component}, G=PC{self.pca_g_component}, B=PC{self.pca_b_component}")
            self._update_display()

    def _change_pca_g(self, val):
        """Change PCA G component."""
        self.pca_g_component = int(val)
        if self.current_mode == ViewMode.PCA and self.pca_rgb_mode:
            print(f"RGB: R=PC{self.pca_r_component}, G=PC{self.pca_g_component}, B=PC{self.pca_b_component}")
            self._update_display()

    def _change_pca_b(self, val):
        """Change PCA B component."""
        self.pca_b_component = int(val)
        if self.current_mode == ViewMode.PCA and self.pca_rgb_mode:
            print(f"RGB: R=PC{self.pca_r_component}, G=PC{self.pca_g_component}, B=PC{self.pca_b_component}")
            self._update_display()

    def _toggle_landmark_category(self, label):
        """Toggle landmark category visibility."""
        if self.current_mode == ViewMode.LANDMARKS:
            print(f"Toggled landmark category: {label}")
            self._update_display()

    def _clear_overlays(self, event):
        """Clear all overlays."""
        self.current_similarities = {}
        self.clicked_point = None
        print("Cleared all overlays")
        self._update_display()

    def show(self):
        """Display the interactive viewer."""
        plt.show()

    def get_pca_summary(self):
        """Get PCA component summary if available."""
        if self.pca_fitted:
            return self.pca_analyzer.get_component_summary()
        return None

    def save_results(self, output_dir: str):
        """Save current analysis results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save similarity results
        if self.current_similarities:
            np.savez(
                output_path / "similarity_results.npz",
                **{f"similarities_{img_id}": sims
                   for img_id, sims in self.current_similarities.items()}
            )

        # Save PCA results
        if self.pca_fitted:
            pca_data = {
                'components': self.pca_analyzer.components,
                'explained_variance_ratio': self.pca_analyzer.explained_variance_ratio,
                'summary': self.get_pca_summary()
            }
            np.savez(output_path / "pca_results.npz", **pca_data)

        print(f"Results saved to {output_path}")