import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

from common.gps import web_mercator


@dataclass
class Landmark:
    """Represents a single landmark from GeoJSON."""
    geometry_type: str  # 'Point', 'LineString', or 'Polygon'
    coordinates: any  # Format depends on geometry_type
    properties: Dict
    category: str
    name: Optional[str] = None

    @property
    def lat(self) -> Optional[float]:
        """Get latitude for Point geometries."""
        if self.geometry_type == 'Point':
            return self.coordinates[1]
        return None

    @property
    def lon(self) -> Optional[float]:
        """Get longitude for Point geometries."""
        if self.geometry_type == 'Point':
            return self.coordinates[0]
        return None


class LandmarkOverlay:
    def __init__(self, geojson_path: str, zoom_level: int = 20):
        self.geojson_path = Path(geojson_path)
        self.zoom_level = zoom_level
        self.landmarks = []
        self.categories = set()
        self.visible_categories = set()

        # Cache for converted coordinates (keyed by image metadata hash)
        self._coord_cache = {}

        # Color map for different landmark categories
        self.category_colors = {
            'amenity': 'red',
            'highway': 'blue',
            'shop': 'green',
            'tourism': 'orange',
            'leisure': 'purple',
            'building': 'brown',
            'railway': 'pink',
            'natural': 'darkgreen',
            'landuse': 'gray',
            'waterway': 'cyan',
            'barrier': 'black',
            'emergency': 'red',
            'healthcare': 'darkred',
            'office': 'navy',
            'craft': 'olive',
            'unknown': 'lightgray'
        }

        self._load_landmarks()

    def _load_landmarks(self):
        """Load landmarks from v3.geojson file."""
        print(f"Loading landmarks from {self.geojson_path}...")

        with open(self.geojson_path, 'r') as f:
            geojson_data = json.load(f)

        if geojson_data['type'] != 'FeatureCollection':
            raise ValueError("Expected FeatureCollection in GeoJSON")

        landmarks_loaded = 0
        geometry_counts = {'Point': 0, 'LineString': 0, 'Polygon': 0, 'MultiPolygon': 0}

        for feature in geojson_data['features']:
            if feature['type'] != 'Feature':
                continue

            geometry = feature['geometry']
            geom_type = geometry['type']

            # Support Point, LineString, Polygon, and MultiPolygon
            if geom_type not in ['Point', 'LineString', 'Polygon', 'MultiPolygon']:
                continue

            properties = feature['properties']
            coordinates = geometry['coordinates']

            # Determine category from properties
            category = self._determine_category(properties)
            self.categories.add(category)

            # Extract name if available
            name = properties.get('name', None)

            # For MultiPolygon, convert to multiple Polygon landmarks
            if geom_type == 'MultiPolygon':
                for poly_coords in coordinates:
                    landmark = Landmark(
                        geometry_type='Polygon',
                        coordinates=poly_coords,
                        properties=properties,
                        category=category,
                        name=name
                    )
                    self.landmarks.append(landmark)
                    landmarks_loaded += 1
                    geometry_counts['MultiPolygon'] += 1
            else:
                landmark = Landmark(
                    geometry_type=geom_type,
                    coordinates=coordinates,
                    properties=properties,
                    category=category,
                    name=name
                )
                self.landmarks.append(landmark)
                landmarks_loaded += 1
                geometry_counts[geom_type] += 1

        print(f"Loaded {landmarks_loaded} landmarks with {len(self.categories)} categories")
        print(f"Geometry types: {geometry_counts}")
        print(f"Categories found: {sorted(self.categories)}")

        # Initially show all categories
        self.visible_categories = self.categories.copy()

    def _determine_category(self, properties: Dict) -> str:
        """Determine landmark category from GeoJSON properties."""
        # Priority order for category assignment
        category_fields = ['amenity', 'highway', 'shop', 'tourism', 'leisure',
                         'building', 'railway', 'natural', 'landuse', 'waterway',
                         'barrier', 'emergency', 'healthcare', 'office', 'craft']

        for field in category_fields:
            if field in properties and properties[field] is not None:
                return field

        return 'unknown'

    def filter_landmarks(self, categories: Optional[Set[str]] = None,
                        bbox: Optional[Tuple[float, float, float, float]] = None) -> List[Landmark]:
        """
        Filter landmarks by category and/or bounding box.

        Args:
            categories: Set of categories to include (None = all visible)
            bbox: Bounding box as (min_lat, min_lon, max_lat, max_lon)

        Returns:
            List of filtered landmarks
        """
        if categories is None:
            categories = self.visible_categories

        filtered = []
        for landmark in self.landmarks:
            # Category filter
            if landmark.category not in categories:
                continue

            # Bounding box filter
            if bbox is not None:
                min_lat, min_lon, max_lat, max_lon = bbox

                # Check if any part of the geometry intersects the bbox
                intersects = False
                if landmark.geometry_type == 'Point':
                    lon, lat = landmark.coordinates
                    intersects = (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon)
                elif landmark.geometry_type == 'LineString':
                    # Check if any point on the line is within bbox
                    for lon, lat in landmark.coordinates:
                        if (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                            intersects = True
                            break
                elif landmark.geometry_type == 'Polygon':
                    # Check if any point of exterior ring is within bbox
                    exterior_ring = landmark.coordinates[0]
                    for lon, lat in exterior_ring:
                        if (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                            intersects = True
                            break

                if not intersects:
                    continue

            filtered.append(landmark)

        return filtered

    def _get_cache_key(self, image_metadata: Dict) -> str:
        """Create a cache key from image metadata."""
        wm_x = image_metadata.get('web_mercator_x', 0)
        wm_y = image_metadata.get('web_mercator_y', 0)
        zoom = image_metadata.get('zoom_level', self.zoom_level)
        return f"{wm_x}_{wm_y}_{zoom}"

    def _latlon_to_image_coords(self, lat: float, lon: float,
                                   image_metadata: Dict,
                                   image_shape: Optional[Tuple[int, int]] = None) -> Tuple[float, float]:
        """Convert a single lat/lon to image pixel coordinates.

        Args:
            lat, lon: Coordinates to convert
            image_metadata: Must contain web_mercator_x, web_mercator_y (CENTER of image)
            image_shape: (height, width) of image for offset calculation
        """
        image_wm_x = image_metadata['web_mercator_x']
        image_wm_y = image_metadata['web_mercator_y']
        image_zoom = image_metadata.get('zoom_level', self.zoom_level)

        # Convert lat/lon to web mercator
        lm_wm_y, lm_wm_x = web_mercator.latlon_to_pixel_coords(lat, lon, image_zoom)

        # Calculate relative position from image CENTER
        rel_x = lm_wm_x - image_wm_x
        rel_y = lm_wm_y - image_wm_y

        # Convert from center-relative to top-left-relative (image pixel coords)
        if image_shape is not None:
            height, width = image_shape
            img_x = rel_x + width / 2
            img_y = rel_y + height / 2
        else:
            # Fallback if no shape provided (backward compatibility)
            img_x = rel_x
            img_y = rel_y

        return img_x, img_y

    def _convert_all_landmarks_to_image_coords(self, image_metadata: Dict,
                                               image_shape: Optional[Tuple[int, int]] = None) -> Dict[int, any]:
        """
        Convert ALL landmarks to image coordinates for a given image.
        Returns a dict keyed by landmark id.
        """
        coords_dict = {}
        for landmark in self.landmarks:
            if landmark.geometry_type == 'Point':
                lon, lat = landmark.coordinates
                img_x, img_y = self._latlon_to_image_coords(lat, lon, image_metadata, image_shape)
                coords_dict[id(landmark)] = (img_x, img_y)

            elif landmark.geometry_type == 'LineString':
                line_coords = []
                for lon, lat in landmark.coordinates:
                    img_x, img_y = self._latlon_to_image_coords(lat, lon, image_metadata, image_shape)
                    line_coords.append((img_x, img_y))
                coords_dict[id(landmark)] = line_coords

            elif landmark.geometry_type == 'Polygon':
                polygon_rings = []
                for ring in landmark.coordinates:  # exterior ring + holes
                    ring_coords = []
                    for lon, lat in ring:
                        img_x, img_y = self._latlon_to_image_coords(lat, lon, image_metadata, image_shape)
                        ring_coords.append((img_x, img_y))
                    polygon_rings.append(ring_coords)
                coords_dict[id(landmark)] = polygon_rings

        return coords_dict

    def landmarks_to_image_coords(self, landmarks: List[Landmark],
                                image_metadata: Dict,
                                image_shape: Optional[Tuple[int, int]] = None,
                                use_cache: bool = True) -> List[Tuple[Landmark, any]]:
        """
        Convert landmark coordinates to image pixel coordinates.

        Args:
            landmarks: List of landmarks to convert
            image_metadata: Metadata containing image coordinate info (web_mercator_x/y are CENTER)
            image_shape: (height, width) of image for proper offset calculation
            use_cache: Whether to use cached coordinates (default True)

        Returns:
            List of (landmark, coords) tuples where coords format depends on geometry type:
            - Point: (x, y)
            - LineString: [(x1, y1), (x2, y2), ...]
            - Polygon: [[(x1, y1), (x2, y2), ...], [...]]  (exterior + holes)
        """
        if 'web_mercator_x' not in image_metadata or 'web_mercator_y' not in image_metadata:
            raise ValueError("Image metadata must contain web_mercator coordinates")

        # Ensure cache is populated for this image
        cache_key = self._get_cache_key(image_metadata)
        if use_cache:
            if cache_key not in self._coord_cache:
                # Populate cache with ALL landmarks for this image
                self._coord_cache[cache_key] = self._convert_all_landmarks_to_image_coords(
                    image_metadata, image_shape)

            # Get coordinates from cache for requested landmarks
            coords_dict = self._coord_cache[cache_key]
            coords_in_image = [(lm, coords_dict[id(lm)]) for lm in landmarks if id(lm) in coords_dict]
        else:
            # No cache - convert on the fly
            coords_in_image = []
            for landmark in landmarks:
                if landmark.geometry_type == 'Point':
                    lon, lat = landmark.coordinates
                    img_x, img_y = self._latlon_to_image_coords(lat, lon, image_metadata, image_shape)
                    coords_in_image.append((landmark, (img_x, img_y)))

                elif landmark.geometry_type == 'LineString':
                    line_coords = []
                    for lon, lat in landmark.coordinates:
                        img_x, img_y = self._latlon_to_image_coords(lat, lon, image_metadata, image_shape)
                        line_coords.append((img_x, img_y))
                    coords_in_image.append((landmark, line_coords))

                elif landmark.geometry_type == 'Polygon':
                    polygon_rings = []
                    for ring in landmark.coordinates:  # exterior ring + holes
                        ring_coords = []
                        for lon, lat in ring:
                            img_x, img_y = self._latlon_to_image_coords(lat, lon, image_metadata, image_shape)
                            ring_coords.append((img_x, img_y))
                        polygon_rings.append(ring_coords)
                    coords_in_image.append((landmark, polygon_rings))

        return coords_in_image

    def overlay_on_image(self, ax: plt.Axes, image_metadata: Dict,
                        image_shape: Tuple[int, int],
                        categories: Optional[Set[str]] = None,
                        show_labels: bool = True) -> None:
        """
        Overlay landmarks on a matplotlib axes.

        Args:
            ax: Matplotlib axes to draw on
            image_metadata: Image metadata with coordinate info
            image_shape: (height, width) of the image
            categories: Categories to display (None = all visible)
            show_labels: Whether to show landmark labels
        """
        height, width = image_shape

        # Create bounding box for the image area
        if 'lat' in image_metadata and 'lon' in image_metadata:
            # Estimate image bounds based on typical satellite image coverage
            # This is approximate - adjust based on actual image coverage
            lat_center = image_metadata['lat']
            lon_center = image_metadata['lon']

            # Rough estimate: ~100m coverage for satellite images
            lat_delta = 0.001  # ~111m per degree latitude
            lon_delta = 0.001  # varies by latitude, but approximate

            bbox = (lat_center - lat_delta, lon_center - lon_delta,
                   lat_center + lat_delta, lon_center + lon_delta)
        else:
            bbox = None

        # Filter landmarks
        filtered_landmarks = self.filter_landmarks(categories, bbox)

        # Convert to image coordinates
        try:
            landmark_coords = self.landmarks_to_image_coords(filtered_landmarks, image_metadata, image_shape)
        except Exception as e:
            print(f"Error converting landmark coordinates: {e}")
            return

        # Plot landmarks
        for landmark, coords in landmark_coords:
            # Get color for category
            color = self.category_colors.get(landmark.category, 'lightgray')

            if landmark.geometry_type == 'Point':
                img_x, img_y = coords
                # Check if landmark is within image bounds
                if not (0 <= img_x < width and 0 <= img_y < height):
                    continue

                # Draw landmark marker
                marker_size = 8
                ax.scatter(img_x, img_y, c=color, s=marker_size**2, alpha=0.8,
                          edgecolors='white', linewidths=1, marker='o')

                # Add label if requested
                if show_labels and landmark.name:
                    ax.annotate(landmark.name,
                              (img_x, img_y),
                              xytext=(5, 5),
                              textcoords='offset points',
                              fontsize=6,
                              bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7),
                              color='white' if color in ['black', 'darkred', 'navy'] else 'black')

            elif landmark.geometry_type == 'LineString':
                # Draw line
                if len(coords) < 2:
                    continue

                # Check if any point is within image bounds
                in_bounds = any(0 <= x < width and 0 <= y < height for x, y in coords)
                if not in_bounds:
                    continue

                xs, ys = zip(*coords)
                ax.plot(xs, ys, color=color, linewidth=2, alpha=0.7)

                # Add label at midpoint if requested
                if show_labels and landmark.name:
                    mid_idx = len(coords) // 2
                    mid_x, mid_y = coords[mid_idx]
                    ax.annotate(landmark.name,
                              (mid_x, mid_y),
                              xytext=(5, 5),
                              textcoords='offset points',
                              fontsize=6,
                              bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7),
                              color='white' if color in ['black', 'darkred', 'navy'] else 'black')

            elif landmark.geometry_type == 'Polygon':
                # Draw polygon (exterior ring + holes)
                if len(coords) == 0 or len(coords[0]) < 3:
                    continue

                # Check if any point is within image bounds
                exterior_ring = coords[0]
                in_bounds = any(0 <= x < width and 0 <= y < height for x, y in exterior_ring)
                if not in_bounds:
                    continue

                # Draw exterior ring as filled polygon
                poly_patch = patches.Polygon(exterior_ring, closed=True,
                                            facecolor=color, edgecolor=color,
                                            alpha=0.3, linewidth=1.5)
                ax.add_patch(poly_patch)

                # Draw holes if any
                for hole in coords[1:]:
                    if len(hole) >= 3:
                        hole_patch = patches.Polygon(hole, closed=True,
                                                    facecolor='white', edgecolor=color,
                                                    alpha=0.8, linewidth=1)
                        ax.add_patch(hole_patch)

                # Add label at centroid if requested
                if show_labels and landmark.name:
                    # Calculate centroid of exterior ring
                    xs, ys = zip(*exterior_ring)
                    centroid_x = sum(xs) / len(xs)
                    centroid_y = sum(ys) / len(ys)
                    ax.annotate(landmark.name,
                              (centroid_x, centroid_y),
                              xytext=(0, 0),
                              textcoords='offset points',
                              fontsize=6,
                              bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7),
                              color='white' if color in ['black', 'darkred', 'navy'] else 'black',
                              ha='center')

    def get_category_summary(self) -> Dict[str, int]:
        """Get count of landmarks by category."""
        summary = {}
        for landmark in self.landmarks:
            summary[landmark.category] = summary.get(landmark.category, 0) + 1
        return summary

    def set_visible_categories(self, categories: Set[str]) -> None:
        """Set which categories should be visible."""
        self.visible_categories = categories.intersection(self.categories)

    def toggle_category(self, category: str) -> None:
        """Toggle visibility of a category."""
        if category in self.visible_categories:
            self.visible_categories.remove(category)
        else:
            self.visible_categories.add(category)

    def get_landmark_info(self, x: int, y: int, image_metadata: Dict,
                         image_shape: Optional[Tuple[int, int]] = None,
                         tolerance: int = 10) -> Optional[Landmark]:
        """
        Get landmark info for a clicked position.

        Args:
            x, y: Pixel coordinates of click
            image_metadata: Image metadata
            image_shape: (height, width) of image for proper coordinate conversion
            tolerance: Pixel tolerance for click detection

        Returns:
            Landmark object if found within tolerance, None otherwise
        """
        try:
            landmark_coords = self.landmarks_to_image_coords(
                self.filter_landmarks(), image_metadata, image_shape)

            for landmark, coords in landmark_coords:
                if landmark.geometry_type == 'Point':
                    lm_x, lm_y = coords
                    distance = np.sqrt((x - lm_x)**2 + (y - lm_y)**2)
                    if distance <= tolerance:
                        return landmark

                elif landmark.geometry_type == 'LineString':
                    # Check if click is near any line segment
                    for i in range(len(coords) - 1):
                        x1, y1 = coords[i]
                        x2, y2 = coords[i + 1]
                        # Calculate distance from point to line segment
                        dist = self._point_to_segment_distance(x, y, x1, y1, x2, y2)
                        if dist <= tolerance:
                            return landmark

                elif landmark.geometry_type == 'Polygon':
                    # Check if click is inside polygon or near boundary
                    exterior_ring = coords[0]
                    if self._point_in_polygon(x, y, exterior_ring):
                        return landmark
                    # Also check if near boundary
                    for i in range(len(exterior_ring)):
                        x1, y1 = exterior_ring[i]
                        x2, y2 = exterior_ring[(i + 1) % len(exterior_ring)]
                        dist = self._point_to_segment_distance(x, y, x1, y1, x2, y2)
                        if dist <= tolerance:
                            return landmark

        except Exception as e:
            print(f"Error finding landmark at ({x}, {y}): {e}")

        return None

    def _point_to_segment_distance(self, px: float, py: float,
                                   x1: float, y1: float,
                                   x2: float, y2: float) -> float:
        """Calculate distance from point to line segment."""
        # Vector from segment start to point
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)

        # Parameter t for closest point on line segment
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))

        # Closest point on segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

    def _point_in_polygon(self, x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
        """Check if point is inside polygon using ray casting algorithm."""
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def create_legend(self, ax: plt.Axes) -> None:
        """Create a legend for landmark categories."""
        legend_elements = []
        for category in sorted(self.visible_categories):
            if category in self.category_colors:
                color = self.category_colors[category]
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=color, markersize=8, label=category.title())
                )

        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1),
                     fontsize=8, framealpha=0.9)