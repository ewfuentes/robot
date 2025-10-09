import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, CheckButtons, TextBox
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import math

from experimental.overhead_matching.swag.data.vigor_dataset import VigorDataset, VigorDatasetConfig, SampleMode


class VigorBrowser:
    def __init__(self, dataset_path: str, config: Optional[VigorDatasetConfig] = None):
        self.dataset_path = Path(dataset_path)

        # Create default config if none provided
        if config is None:
            config = VigorDatasetConfig(
                satellite_tensor_cache_info=None,
                panorama_tensor_cache_info=None,
                sample_mode=SampleMode.NEAREST,
                should_load_images=True
            )

        # Load VIGOR dataset
        self.dataset = VigorDataset(self.dataset_path, config)
        self.sat_dataset = self.dataset.get_sat_patch_view()
        self.pano_dataset = self.dataset.get_pano_view()

        # Browser state
        self.selected_images = {}  # {(type, idx): (image_array, metadata)}
        self.current_page = 0
        self.items_per_page = 12
        self.image_type_filter = {'satellite': True, 'panorama': True}

        # GPS-based browsing
        self.reference_lat = None
        self.reference_lon = None
        self.sort_by_distance = False

        # UI components
        self.fig = None
        self.axes = None
        self.buttons = {}
        self.checkboxes = None
        self.page_text = None
        self.lat_textbox = None
        self.lon_textbox = None

        # Data to return when selection is complete
        self.selected_data = None
        self.selection_complete = False

    def show(self) -> Optional[Dict]:
        """Show the browser interface and return selected image data.

        Returns:
            Dict of selected image data, or None if no selection was made
        """
        self._setup_ui()
        plt.show()
        return self.selected_data

    def _setup_ui(self):
        """Setup the matplotlib UI for browsing."""
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle('VIGOR Dataset Browser - Select Images for Similarity Analysis', fontsize=16)

        # Create grid layout using GridSpec for better control
        # 11 rows total: 9 for images (3 rows × 3 height each), 2 for controls
        gs = GridSpec(11, 8, figure=self.fig, hspace=0.4, wspace=0.3)

        # Create image display axes (3 rows × 4 cols = 12 images)
        # Each image spans 3 rows in height, 2 columns in width
        self.axes = []
        for row in range(3):
            for col in range(4):
                ax = self.fig.add_subplot(gs[row * 3:(row + 1) * 3, col * 2:(col + 1) * 2])
                ax.set_xticks([])
                ax.set_yticks([])
                self.axes.append(ax)

        # Control panel area (last row)
        control_row = 9

        # Previous/Next page buttons
        ax_prev = self.fig.add_subplot(gs[control_row, 0])
        self.buttons['prev'] = Button(ax_prev, 'Previous')
        self.buttons['prev'].on_clicked(self._prev_page)

        ax_next = self.fig.add_subplot(gs[control_row, 1])
        self.buttons['next'] = Button(ax_next, 'Next')
        self.buttons['next'].on_clicked(self._next_page)

        # Page display
        ax_page = self.fig.add_subplot(gs[control_row, 2])
        ax_page.text(0.5, 0.5, f'Page {self.current_page + 1}',
                    transform=ax_page.transAxes, ha='center', va='center')
        ax_page.set_xticks([])
        ax_page.set_yticks([])

        # Image type filter checkboxes
        ax_checks = self.fig.add_subplot(gs[control_row, 3:5])
        self.checkboxes = CheckButtons(ax_checks, ['Satellite', 'Panorama'],
                                     [self.image_type_filter['satellite'],
                                      self.image_type_filter['panorama']])
        self.checkboxes.on_clicked(self._toggle_image_type)

        # Load selected button
        ax_load = self.fig.add_subplot(gs[control_row, 5])
        self.buttons['load'] = Button(ax_load, 'Load Selected')
        self.buttons['load'].on_clicked(self._load_selected)

        # Clear selection button
        ax_clear = self.fig.add_subplot(gs[control_row, 6])
        self.buttons['clear'] = Button(ax_clear, 'Clear All')
        self.buttons['clear'].on_clicked(self._clear_selection)

        # Selection info
        ax_info = self.fig.add_subplot(gs[control_row, 7])
        ax_info.text(0.5, 0.5, f'Selected: {len(self.selected_images)}',
                    transform=ax_info.transAxes, ha='center', va='center')
        ax_info.set_xticks([])
        ax_info.set_yticks([])
        self.info_ax = ax_info

        # GPS search controls (second control row)
        gps_row = control_row + 1

        # GPS label
        ax_gps_label = self.fig.add_subplot(gs[gps_row, 0])
        ax_gps_label.text(0.5, 0.5, 'GPS Search:',
                         transform=ax_gps_label.transAxes, ha='center', va='center', fontsize=9)
        ax_gps_label.set_xticks([])
        ax_gps_label.set_yticks([])

        # Latitude textbox
        ax_lat = self.fig.add_subplot(gs[gps_row, 1:3])
        self.lat_textbox = TextBox(ax_lat, 'Lat:', initial='')

        # Longitude textbox
        ax_lon = self.fig.add_subplot(gs[gps_row, 3:5])
        self.lon_textbox = TextBox(ax_lon, 'Lon:', initial='')

        # Find Near button
        ax_find = self.fig.add_subplot(gs[gps_row, 5])
        self.buttons['find'] = Button(ax_find, 'Find Near')
        self.buttons['find'].on_clicked(self._find_near_location)

        # Reset sort button
        ax_reset = self.fig.add_subplot(gs[gps_row, 6])
        self.buttons['reset'] = Button(ax_reset, 'Reset')
        self.buttons['reset'].on_clicked(self._reset_sort)

        # Distance info
        ax_dist_info = self.fig.add_subplot(gs[gps_row, 7])
        ax_dist_info.text(0.5, 0.5, '',
                         transform=ax_dist_info.transAxes, ha='center', va='center', fontsize=8)
        ax_dist_info.set_xticks([])
        ax_dist_info.set_yticks([])
        self.dist_info_ax = ax_dist_info

        # Initial display
        self._update_display()

        # Connect click events
        self.fig.canvas.mpl_connect('button_press_event', self._on_image_click)

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance in meters between two GPS coordinates."""
        from math import radians, sin, cos, sqrt, atan2

        R = 6371000  # Earth radius in meters

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        return R * c

    def _get_current_items(self) -> List[Tuple[str, int, np.ndarray, Dict]]:
        """Get items for current page with filtering (lazy loading)."""
        # Build a lightweight list of just (type, dataset_idx) tuples - no metadata yet!
        all_item_indices = []

        if self.image_type_filter['satellite']:
            all_item_indices.extend([('satellite', i) for i in range(len(self.sat_dataset))])

        if self.image_type_filter['panorama']:
            all_item_indices.extend([('panorama', i) for i in range(len(self.pano_dataset))])

        # Sort by distance if GPS reference is set
        if self.sort_by_distance and self.reference_lat is not None and self.reference_lon is not None:
            # Get metadata references
            sat_metadata = self.dataset._satellite_metadata if hasattr(self.dataset, '_satellite_metadata') else None
            pano_metadata = self.dataset._panorama_metadata if hasattr(self.dataset, '_panorama_metadata') else None

            def get_distance(item):
                img_type, idx = item
                metadata_df = sat_metadata if img_type == 'satellite' else pano_metadata

                if metadata_df is not None and idx < len(metadata_df):
                    # Access lat/lon directly from dataframe without converting to dict
                    try:
                        lat = metadata_df.iloc[idx]['lat']
                        lon = metadata_df.iloc[idx]['lon']
                        return self._haversine_distance(
                            self.reference_lat, self.reference_lon, lat, lon
                        )
                    except (KeyError, TypeError):
                        pass
                return float('inf')  # Put items without GPS at the end

            all_item_indices.sort(key=get_distance)

        # Calculate pagination range
        start_idx = self.current_page * self.items_per_page
        end_idx = start_idx + self.items_per_page

        # Get only the indices for current page
        page_indices = all_item_indices[start_idx:end_idx]

        # Load only the images and metadata for the current page
        sat_metadata = self.dataset._satellite_metadata if hasattr(self.dataset, '_satellite_metadata') else None
        pano_metadata = self.dataset._panorama_metadata if hasattr(self.dataset, '_panorama_metadata') else None

        items = []
        for img_type, idx in page_indices:
            try:
                # Get metadata for this specific item
                metadata_df = sat_metadata if img_type == 'satellite' else pano_metadata
                if metadata_df is not None and idx < len(metadata_df):
                    metadata = metadata_df.iloc[idx].to_dict()
                else:
                    metadata = {}

                # Load image
                if img_type == 'satellite':
                    sample = self.sat_dataset[idx]
                    img_tensor = sample.satellite if hasattr(sample, 'satellite') else sample.image
                else:  # panorama
                    sample = self.pano_dataset[idx]
                    img_tensor = sample.panorama if hasattr(sample, 'panorama') else sample.image

                img_array = self._tensor_to_array(img_tensor)
                items.append((img_type, idx, img_array, metadata))
            except Exception as e:
                print(f"Error loading {img_type} image {idx}: {e}")
                continue

        return items

    def _tensor_to_array(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert torch tensor to numpy array for display."""
        if tensor.dim() == 4:  # Batch dimension
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3:  # Channel first
            tensor = tensor.permute(1, 2, 0)

        array = tensor.cpu().numpy()

        # Normalize to [0, 1] if needed
        if array.max() > 1.0:
            array = array / 255.0

        # Ensure valid range
        array = np.clip(array, 0, 1)

        return array

    def _update_display(self):
        """Update the display with current page items."""
        # Clear all axes
        for ax in self.axes:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

        # Get current items
        items = self._get_current_items()

        # Display items
        for i, (img_type, idx, img_array, metadata) in enumerate(items):
            if i >= len(self.axes):
                break

            ax = self.axes[i]

            # Display image
            ax.imshow(img_array)

            # Create title with metadata
            title_parts = [f"{img_type.title()} {idx}"]
            if 'lat' in metadata and 'lon' in metadata:
                title_parts.append(f"({metadata['lat']:.4f}, {metadata['lon']:.4f})")

                # Add distance if in distance mode
                if self.sort_by_distance and self.reference_lat is not None:
                    dist = self._haversine_distance(
                        self.reference_lat, self.reference_lon,
                        metadata['lat'], metadata['lon']
                    )
                    if dist < 1000:
                        title_parts.append(f"{dist:.0f}m")
                    else:
                        title_parts.append(f"{dist/1000:.1f}km")

            title = '\n'.join(title_parts)
            ax.set_title(title, fontsize=8)

            # Highlight if selected
            key = (img_type, idx)
            if key in self.selected_images:
                # Add green border for selected images
                rect = patches.Rectangle((0, 0), img_array.shape[1], img_array.shape[0],
                                       linewidth=3, edgecolor='green', facecolor='none')
                ax.add_patch(rect)

            # Store item info for click handling
            ax._vigor_item = (img_type, idx, img_array, metadata)

        # Update selection counter
        self.info_ax.clear()
        self.info_ax.text(0.5, 0.5, f'Selected: {len(self.selected_images)}',
                         transform=self.info_ax.transAxes, ha='center', va='center')
        self.info_ax.set_xticks([])
        self.info_ax.set_yticks([])

        plt.draw()

    def _update_selection_border(self, ax, img_type, idx, img_array):
        """Update just the selection border for a single image without reloading."""
        # Remove any existing patches (borders)
        for patch in ax.patches[:]:
            patch.remove()

        # Add green border if selected
        key = (img_type, idx)
        if key in self.selected_images:
            rect = patches.Rectangle((0, 0), img_array.shape[1], img_array.shape[0],
                                   linewidth=3, edgecolor='green', facecolor='none')
            ax.add_patch(rect)

    def _on_image_click(self, event):
        """Handle clicks on images to select/deselect them."""
        if event.inaxes in self.axes and hasattr(event.inaxes, '_vigor_item'):
            img_type, idx, img_array, metadata = event.inaxes._vigor_item
            key = (img_type, idx)

            if key in self.selected_images:
                # Deselect
                del self.selected_images[key]
            else:
                # Select
                self.selected_images[key] = (img_array, metadata)

            # Just update the selection border on this specific image, not the whole display
            self._update_selection_border(event.inaxes, img_type, idx, img_array)

            # Update selection counter
            self.info_ax.clear()
            self.info_ax.text(0.5, 0.5, f'Selected: {len(self.selected_images)}',
                             transform=self.info_ax.transAxes, ha='center', va='center')
            self.info_ax.set_xticks([])
            self.info_ax.set_yticks([])

            plt.draw()

    def _prev_page(self, event):
        """Go to previous page."""
        if self.current_page > 0:
            self.current_page -= 1
            self._update_display()

    def _next_page(self, event):
        """Go to next page."""
        total_items = 0
        if self.image_type_filter['satellite']:
            total_items += len(self.sat_dataset)
        if self.image_type_filter['panorama']:
            total_items += len(self.pano_dataset)

        max_page = math.ceil(total_items / self.items_per_page) - 1
        if self.current_page < max_page:
            self.current_page += 1
            self._update_display()

    def _toggle_image_type(self, label):
        """Toggle image type filter."""
        if label == 'Satellite':
            self.image_type_filter['satellite'] = not self.image_type_filter['satellite']
        elif label == 'Panorama':
            self.image_type_filter['panorama'] = not self.image_type_filter['panorama']

        self.current_page = 0  # Reset to first page when filter changes
        self._update_display()

    def _clear_selection(self, event):
        """Clear all selected images."""
        self.selected_images = {}
        self._update_display()

    def _load_selected(self, event):
        """Load selected images for similarity analysis."""
        if not self.selected_images:
            print("No images selected!")
            return

        print(f"Loading {len(self.selected_images)} selected images...")

        # Convert selected images to format expected by similarity viewer
        image_data = {}
        for (img_type, idx), (img_array, metadata) in self.selected_images.items():
            # Create a unique identifier for each image
            image_id = f"{img_type}_{idx}"
            image_data[image_id] = {
                'array': img_array,
                'metadata': metadata,
                'type': img_type,
                'dataset_index': idx
            }

        # Store the data to be returned
        self.selected_data = image_data
        self.selection_complete = True

        # Close the browser window - show() will return after this
        plt.close(self.fig)

    def _find_near_location(self, event):
        """Find images near the specified GPS coordinates."""
        try:
            lat_str = self.lat_textbox.text.strip()
            lon_str = self.lon_textbox.text.strip()

            if not lat_str or not lon_str:
                print("Please enter both latitude and longitude")
                return

            self.reference_lat = float(lat_str)
            self.reference_lon = float(lon_str)
            self.sort_by_distance = True
            self.current_page = 0  # Reset to first page

            # Update distance info display
            self.dist_info_ax.clear()
            self.dist_info_ax.text(0.5, 0.5, f'Sorted by distance',
                                  transform=self.dist_info_ax.transAxes, ha='center', va='center', fontsize=8)
            self.dist_info_ax.set_xticks([])
            self.dist_info_ax.set_yticks([])

            print(f"Sorting by distance from ({self.reference_lat:.4f}, {self.reference_lon:.4f})")
            self._update_display()

        except ValueError:
            print("Invalid GPS coordinates. Please enter valid numbers.")

    def _reset_sort(self, event):
        """Reset to default ordering (no distance sorting)."""
        self.reference_lat = None
        self.reference_lon = None
        self.sort_by_distance = False
        self.current_page = 0  # Reset to first page

        # Clear distance info display
        self.dist_info_ax.clear()
        self.dist_info_ax.set_xticks([])
        self.dist_info_ax.set_yticks([])

        print("Reset to default ordering")
        self._update_display()

    def get_selected_image_data(self) -> Dict:
        """Get the selected image data for external use."""
        image_data = {}
        for (img_type, idx), (img_array, metadata) in self.selected_images.items():
            image_id = f"{img_type}_{idx}"
            image_data[image_id] = {
                'array': img_array,
                'metadata': metadata,
                'type': img_type,
                'dataset_index': idx
            }
        return image_data
