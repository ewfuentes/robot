import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional

from experimental.overhead_matching.swag.scripts.image_similarity.similarity_engine import ImageSimilarityEngine


class InteractiveImageViewer:
    def __init__(self, image_paths: List[str], figsize: tuple = (16, 8)):
        self.image_paths = image_paths
        self.num_images = len(image_paths)
        self.similarity_engine = ImageSimilarityEngine()

        # Load all images
        self.images = {}
        for path in image_paths:
            img = Image.open(path).convert('RGB')
            self.images[path] = np.array(img)

        # Set up the matplotlib figure
        self.fig, self.axes = plt.subplots(1, self.num_images, figsize=figsize)
        if self.num_images == 1:
            self.axes = [self.axes]

        # Initialize display
        self.imshows = []
        self.overlays = []
        self.clicked_point = None

        for i, (path, ax) in enumerate(zip(image_paths, self.axes)):
            # Display the image
            imshow = ax.imshow(self.images[path])
            self.imshows.append(imshow)

            # Initialize empty overlay for similarity heatmap
            overlay = ax.imshow(np.zeros_like(self.images[path][:, :, 0]),
                              alpha=0.0, cmap='hot', vmin=0, vmax=1)
            self.overlays.append(overlay)

            ax.set_title(f"Image {i+1}: {Path(path).name}")
            ax.axis('off')

            # Connect click event
            ax.figure.canvas.mpl_connect('button_press_event', self.on_click)

        # Add control buttons
        self.setup_controls()

        # Store current similarity results
        self.current_similarities = {}

        plt.tight_layout()

    def setup_controls(self):
        """Add control buttons to the interface."""
        # Add clear button
        ax_clear = plt.axes([0.02, 0.02, 0.1, 0.04])
        self.clear_button = Button(ax_clear, 'Clear')
        self.clear_button.on_clicked(self.clear_overlays)

        # Add colorbar axis
        self.cbar_ax = plt.axes([0.92, 0.15, 0.02, 0.7])
        self.colorbar = None

    def on_click(self, event):
        """Handle mouse click events on images."""
        if event.inaxes is None:
            return

        # Find which image was clicked
        clicked_image_idx = None
        for i, ax in enumerate(self.axes):
            if event.inaxes == ax:
                clicked_image_idx = i
                break

        if clicked_image_idx is None:
            return

        clicked_path = self.image_paths[clicked_image_idx]
        clicked_x = int(event.xdata) if event.xdata is not None else 0
        clicked_y = int(event.ydata) if event.ydata is not None else 0

        print(f"Clicked at ({clicked_x}, {clicked_y}) in {Path(clicked_path).name}")

        # Clear previous click marker
        if self.clicked_point is not None:
            self.clicked_point.remove()

        # Add click marker
        self.clicked_point = self.axes[clicked_image_idx].plot(
            clicked_x, clicked_y, 'r+', markersize=15, markeredgewidth=3
        )[0]

        # Compute similarities
        try:
            self.current_similarities = self.similarity_engine.compute_similarities(
                clicked_path, clicked_x, clicked_y, self.image_paths
            )

            # Update similarity overlays
            self.update_similarity_overlays()

        except Exception as e:
            print(f"Error computing similarities: {e}")

        plt.draw()

    def update_similarity_overlays(self):
        """Update the similarity heatmap overlays on all images."""
        if not self.current_similarities:
            return

        # Find global min/max for consistent color scaling
        all_similarities = np.concatenate(list(self.current_similarities.values()))
        vmin, vmax = all_similarities.min(), all_similarities.max()

        for i, (path, overlay) in enumerate(zip(self.image_paths, self.overlays)):
            if path in self.current_similarities:
                # Convert similarities to heatmap
                heatmap = self.similarity_engine.similarity_to_heatmap(
                    self.current_similarities[path], path
                )

                # Update overlay
                overlay.set_array(heatmap)
                overlay.set_alpha(0.6)
                overlay.set_clim(vmin=vmin, vmax=vmax)

        # Update colorbar
        if self.colorbar is not None:
            self.colorbar.remove()

        if self.overlays:
            self.colorbar = plt.colorbar(
                self.overlays[0], cax=self.cbar_ax, label='Similarity'
            )

    def clear_overlays(self, event=None):
        """Clear all similarity overlays."""
        for overlay in self.overlays:
            overlay.set_alpha(0.0)

        if self.clicked_point is not None:
            self.clicked_point.remove()
            self.clicked_point = None

        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None

        self.current_similarities = {}
        plt.draw()

    def show(self):
        """Display the interactive viewer."""
        plt.show()

    def save_similarity_results(self, output_path: str):
        """Save current similarity results to file."""
        if not self.current_similarities:
            print("No similarity results to save")
            return

        np.savez(
            output_path,
            **{f"similarities_{Path(path).stem}": sims
               for path, sims in self.current_similarities.items()}
        )
        print(f"Similarity results saved to {output_path}")