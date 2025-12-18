"""GPU-accelerated geometry collection and spatial index.

This module provides high-level abstractions for working with collections
of 2D geometries (Points, LineStrings, Polygons, MultiPolygons) on GPU.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import shapely
import torch

from common.geometry import gpu_distance


class GeometryType(IntEnum):
    """Enum for geometry types."""

    POINT = 0
    LINESTRING = 1
    POLYGON = 2
    MULTIPOLYGON = 3


@dataclass
class SpatialIndex:
    """Uniform grid spatial index for fast distance queries.

    Uses CSR (Compressed Sparse Row) format to efficiently store which
    segments/points belong to each grid cell.

    Attributes:
        grid_origin: (2,) tensor - (x_min, y_min) of grid bounds
        cell_size: Grid cell size for hashing particles
        expansion_distance: Distance to expand geometries
        grid_dims: (2,) tensor - (num_cells_x, num_cells_y)
        cell_segment_indices: (K,) tensor - segment indices sorted by cell
        cell_offsets: (num_cells+1,) tensor - CSR row pointers for segments
        cell_point_indices: (P',) tensor - point indices sorted by cell
        cell_point_offsets: (num_cells+1,) tensor - CSR row pointers for points
    """

    grid_origin: torch.Tensor
    cell_size: float
    expansion_distance: float
    grid_dims: torch.Tensor
    cell_segment_indices: torch.Tensor
    cell_offsets: torch.Tensor
    cell_point_indices: torch.Tensor
    cell_point_offsets: torch.Tensor


@dataclass
class GPUGeometryCollection:
    """GPU-friendly representation of a collection of 2D geometries.

    Supports Point, LineString, Polygon, MultiPolygon.
    All data stored as GPU tensors for efficient batch operations.

    Attributes:
        device: The torch device where tensors are stored.
        num_geometries: Total number of geometries in the collection.
        geometry_types: (G,) tensor of GeometryType enum values.
        segment_starts: (S, 2) tensor of segment start coordinates.
        segment_ends: (S, 2) tensor of segment end coordinates.
        segment_to_geom: (S,) tensor mapping each segment to geometry index.
        point_coords: (P, 2) tensor of point geometry coordinates.
        point_to_geom: (P,) tensor mapping each point to geometry index.
        polygon_vertices: (V, 2) tensor of polygon ring vertices.
        polygon_ranges: (num_polygons, 2) tensor of [start, end) vertex indices.
        polygon_geom_indices: (num_polygons,) tensor mapping to geometry index.
        spatial_index: Optional spatial index for fast distance queries.
    """

    device: torch.device
    num_geometries: int
    geometry_types: torch.Tensor

    # Segment data (for LineStrings and Polygon boundaries)
    segment_starts: torch.Tensor
    segment_ends: torch.Tensor
    segment_to_geom: torch.Tensor

    # Point data
    point_coords: torch.Tensor
    point_to_geom: torch.Tensor

    # Polygon data (for inside/outside tests)
    polygon_vertices: torch.Tensor
    polygon_ranges: torch.Tensor
    polygon_geom_indices: torch.Tensor

    # Spatial index (optional, built on demand)
    spatial_index: SpatialIndex | None = None

    @classmethod
    def from_shapely(
        cls,
        geometries: Sequence[shapely.Geometry],
        device: torch.device = torch.device("cpu"),
    ) -> "GPUGeometryCollection":
        """Convert shapely geometries to GPU representation.

        Args:
            geometries: Sequence of shapely geometry objects.
            device: Target torch device.

        Returns:
            GPUGeometryCollection with all geometries converted.
        """
        num_geometries = len(geometries)
        geometry_types = []

        # Collect segments from all geometries
        segment_starts_list = []
        segment_ends_list = []
        segment_to_geom_list = []

        # Collect points
        point_coords_list = []
        point_to_geom_list = []

        # Collect polygon data
        polygon_vertices_list = []
        polygon_ranges_list = []
        polygon_geom_indices_list = []
        vertex_offset = 0

        for geom_idx, geom in enumerate(geometries):
            geom_type = geom.geom_type

            if geom_type == "Point":
                geometry_types.append(GeometryType.POINT)
                point_coords_list.append([geom.x, geom.y])
                point_to_geom_list.append(geom_idx)

            elif geom_type == "LineString":
                geometry_types.append(GeometryType.LINESTRING)
                coords = list(geom.coords)
                for i in range(len(coords) - 1):
                    segment_starts_list.append(coords[i])
                    segment_ends_list.append(coords[i + 1])
                    segment_to_geom_list.append(geom_idx)

            elif geom_type == "Polygon":
                geometry_types.append(GeometryType.POLYGON)
                _process_polygon(
                    geom,
                    geom_idx,
                    segment_starts_list,
                    segment_ends_list,
                    segment_to_geom_list,
                    polygon_vertices_list,
                    polygon_ranges_list,
                    polygon_geom_indices_list,
                    vertex_offset,
                )
                # Update vertex offset
                ring_coords = list(geom.exterior.coords)[:-1]
                vertex_offset += len(ring_coords)

            elif geom_type == "MultiPolygon":
                geometry_types.append(GeometryType.MULTIPOLYGON)
                for poly in geom.geoms:
                    _process_polygon(
                        poly,
                        geom_idx,
                        segment_starts_list,
                        segment_ends_list,
                        segment_to_geom_list,
                        polygon_vertices_list,
                        polygon_ranges_list,
                        polygon_geom_indices_list,
                        vertex_offset,
                    )
                    ring_coords = list(poly.exterior.coords)[:-1]
                    vertex_offset += len(ring_coords)

            else:
                raise ValueError(f"Unsupported geometry type: {geom_type}")

        # Convert to tensors
        geometry_types_tensor = torch.tensor(
            geometry_types, dtype=torch.long, device=device
        )

        # Segments
        if segment_starts_list:
            segment_starts = torch.tensor(
                segment_starts_list, dtype=torch.float32, device=device
            )
            segment_ends = torch.tensor(
                segment_ends_list, dtype=torch.float32, device=device
            )
            segment_to_geom = torch.tensor(
                segment_to_geom_list, dtype=torch.long, device=device
            )
        else:
            segment_starts = torch.empty((0, 2), dtype=torch.float32, device=device)
            segment_ends = torch.empty((0, 2), dtype=torch.float32, device=device)
            segment_to_geom = torch.empty((0,), dtype=torch.long, device=device)

        # Points
        if point_coords_list:
            point_coords = torch.tensor(
                point_coords_list, dtype=torch.float32, device=device
            )
            point_to_geom = torch.tensor(
                point_to_geom_list, dtype=torch.long, device=device
            )
        else:
            point_coords = torch.empty((0, 2), dtype=torch.float32, device=device)
            point_to_geom = torch.empty((0,), dtype=torch.long, device=device)

        # Polygons
        if polygon_vertices_list:
            polygon_vertices = torch.tensor(
                polygon_vertices_list, dtype=torch.float32, device=device
            )
            polygon_ranges = torch.tensor(
                polygon_ranges_list, dtype=torch.long, device=device
            )
            polygon_geom_indices = torch.tensor(
                polygon_geom_indices_list, dtype=torch.long, device=device
            )
        else:
            polygon_vertices = torch.empty((0, 2), dtype=torch.float32, device=device)
            polygon_ranges = torch.empty((0, 2), dtype=torch.long, device=device)
            polygon_geom_indices = torch.empty((0,), dtype=torch.long, device=device)

        return cls(
            device=device,
            num_geometries=num_geometries,
            geometry_types=geometry_types_tensor,
            segment_starts=segment_starts,
            segment_ends=segment_ends,
            segment_to_geom=segment_to_geom,
            point_coords=point_coords,
            point_to_geom=point_to_geom,
            polygon_vertices=polygon_vertices,
            polygon_ranges=polygon_ranges,
            polygon_geom_indices=polygon_geom_indices,
        )

    def distance_to_points(
        self,
        query_points: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distance from each query point to each geometry.

        For polygons, returns signed distance (negative if inside).

        Args:
            query_points: (Q, 2) tensor of query point coordinates.

        Returns:
            (Q, G) tensor where result[i, j] is the distance from
            query_points[i] to geometry j.
        """
        Q = query_points.shape[0]
        G = self.num_geometries

        # Initialize distances to infinity
        distances = torch.full((Q, G), float("inf"), device=self.device)

        # Handle Point geometries
        if self.point_coords.shape[0] > 0:
            point_distances = gpu_distance.point_to_point_distance(
                query_points.unsqueeze(1),  # (Q, 1, 2)
                self.point_coords.unsqueeze(0),  # (1, P, 2)
            )  # (Q, P)

            # Scatter into correct geometry positions
            for i, geom_idx in enumerate(self.point_to_geom):
                distances[:, geom_idx] = point_distances[:, i]

        # Identify which geometries are polygons (should use signed distance)
        is_polygon_geom = torch.zeros(G, dtype=torch.bool, device=self.device)
        if self.polygon_geom_indices.shape[0] > 0:
            is_polygon_geom[self.polygon_geom_indices.unique()] = True

        # Handle LineString boundary distances (only for non-polygon geometries)
        if self.segment_starts.shape[0] > 0:
            segment_distances = gpu_distance.point_to_segments_min_distance(
                query_points,
                self.segment_starts,
                self.segment_ends,
                self.segment_to_geom,
                G,
            )  # (Q, G)

            # Update only non-polygon geometries with segment distances
            has_segments = torch.zeros(G, dtype=torch.bool, device=self.device)
            has_segments[self.segment_to_geom.unique()] = True
            non_polygon_segments = has_segments & ~is_polygon_geom
            distances[:, non_polygon_segments] = torch.minimum(
                distances[:, non_polygon_segments],
                segment_distances[:, non_polygon_segments],
            )

        # Handle Polygon signed distances
        if self.polygon_ranges.shape[0] > 0:
            signed_distances = gpu_distance.signed_distance_to_polygons(
                query_points, self.polygon_vertices, self.polygon_ranges
            )  # (Q, num_polygon_rings)

            # Map back to geometry indices
            # For MultiPolygons, take minimum absolute distance but preserve sign
            for i, geom_idx in enumerate(self.polygon_geom_indices):
                geom_idx_val = geom_idx.item()
                current = distances[:, geom_idx_val]
                new_dist = signed_distances[:, i]
                # Use absolute value comparison but preserve sign of closer one
                closer_mask = new_dist.abs() < current.abs()
                distances[:, geom_idx_val] = torch.where(closer_mask, new_dist, current)

        return distances

    def build_spatial_index(
        self,
        cell_size: float,
        expansion_distance: float,
        grid_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
        profile: dict | None = None,
    ) -> None:
        """Build uniform grid spatial index for fast distance queries.

        Args:
            cell_size: Grid discretization for hashing particles (e.g., sigma_px)
            expansion_distance: Distance to expand geometries (e.g., 5*sigma_px)
            grid_bounds: Optional (bbox_min, bbox_max) tensors of shape (2,).
                If None, computed from geometry bounding box.
            profile: Optional dict to store detailed timing information.
        """
        # Step 1: Compute or use provided grid bounds
        if grid_bounds is not None:
            bbox_min, bbox_max = grid_bounds
        else:
            # Compute bounds from geometry without creating intermediate tensor
            bbox_min = None
            bbox_max = None

            if self.segment_starts.shape[0] > 0:
                seg_min = torch.minimum(self.segment_starts.min(dim=0).values,
                                       self.segment_ends.min(dim=0).values)
                seg_max = torch.maximum(self.segment_starts.max(dim=0).values,
                                       self.segment_ends.max(dim=0).values)
                bbox_min = seg_min if bbox_min is None else torch.minimum(bbox_min, seg_min)
                bbox_max = seg_max if bbox_max is None else torch.maximum(bbox_max, seg_max)

            if self.point_coords.shape[0] > 0:
                pt_min = self.point_coords.min(dim=0).values
                pt_max = self.point_coords.max(dim=0).values
                bbox_min = pt_min if bbox_min is None else torch.minimum(bbox_min, pt_min)
                bbox_max = pt_max if bbox_max is None else torch.maximum(bbox_max, pt_max)

            if bbox_min is None:
                # Empty collection - create empty index
                grid_origin = torch.zeros(2, dtype=torch.float32, device=self.device)
                grid_dims = torch.ones(2, dtype=torch.long, device=self.device)
                num_cells = 1

                self.spatial_index = SpatialIndex(
                    grid_origin=grid_origin,
                    cell_size=cell_size,
                    expansion_distance=expansion_distance,
                    grid_dims=grid_dims,
                    cell_segment_indices=torch.empty(0, dtype=torch.long, device=self.device),
                    cell_offsets=torch.zeros(num_cells + 1, dtype=torch.long, device=self.device),
                    cell_point_indices=torch.empty(0, dtype=torch.long, device=self.device),
                    cell_point_offsets=torch.zeros(num_cells + 1, dtype=torch.long, device=self.device),
                )
                return

            # Add expansion to computed bounds
            bbox_min = bbox_min - expansion_distance
            bbox_max = bbox_max + expansion_distance

        grid_origin = bbox_min
        grid_dims = torch.ceil((bbox_max - bbox_min) / cell_size).long()
        num_cells = grid_dims[0].item() * grid_dims[1].item()

        # Step 2: Build segment index
        seg_profile = {} if profile is not None else None
        cell_segment_indices, cell_offsets = self._index_segments(
            grid_origin, cell_size, grid_dims, expansion_distance, seg_profile
        )
        if profile is not None:
            profile['segments'] = seg_profile

        # Step 3: Build point index
        pt_profile = {} if profile is not None else None
        cell_point_indices, cell_point_offsets = self._index_points(
            grid_origin, cell_size, grid_dims, expansion_distance, pt_profile
        )
        if profile is not None:
            profile['points'] = pt_profile

        # Step 4: Store index
        self.spatial_index = SpatialIndex(
            grid_origin=grid_origin,
            cell_size=cell_size,
            expansion_distance=expansion_distance,
            grid_dims=grid_dims,
            cell_segment_indices=cell_segment_indices,
            cell_offsets=cell_offsets,
            cell_point_indices=cell_point_indices,
            cell_point_offsets=cell_point_offsets,
        )

    def _index_segments(
        self,
        grid_origin: torch.Tensor,
        cell_size: float,
        grid_dims: torch.Tensor,
        expansion_distance: float,
        profile: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build CSR index for segments.

        Args:
            profile: Optional dict to store timing information

        Returns:
            Tuple of (cell_segment_indices, cell_offsets) in CSR format
        """
        import time

        if self.segment_starts.shape[0] == 0:
            num_cells = grid_dims[0].item() * grid_dims[1].item()
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.zeros(num_cells + 1, dtype=torch.long, device=self.device),
            )

        t0 = time.time()
        # Compute expanded bbox for each segment
        seg_bbox_min = (
            torch.minimum(self.segment_starts, self.segment_ends) - expansion_distance
        )
        seg_bbox_max = (
            torch.maximum(self.segment_starts, self.segment_ends) + expansion_distance
        )
        if profile is not None:
            profile['bbox_compute'] = time.time() - t0

        t0 = time.time()
        # Convert to cell coordinates
        cell_min_unclamped = torch.floor((seg_bbox_min - grid_origin) / cell_size).long()
        cell_max_unclamped = torch.floor((seg_bbox_max - grid_origin) / cell_size).long()

        # Clamp to grid bounds
        zeros = torch.zeros_like(grid_dims)
        grid_max = grid_dims - 1
        cell_min = torch.maximum(cell_min_unclamped, zeros)
        cell_min = torch.minimum(cell_min, grid_max)
        cell_max = torch.maximum(cell_max_unclamped, zeros)
        cell_max = torch.minimum(cell_max, grid_max)

        # Check if segment bbox intersects grid [0, grid_dims-1]
        # Segment intersects if: cell_max_unclamped >= 0 AND cell_min_unclamped < grid_dims
        intersects_grid = (
            (cell_max_unclamped[:, 0] >= 0) &
            (cell_max_unclamped[:, 1] >= 0) &
            (cell_min_unclamped[:, 0] < grid_dims[0]) &
            (cell_min_unclamped[:, 1] < grid_dims[1])
        )
        if profile is not None:
            profile['cell_coords'] = time.time() - t0

        t0 = time.time()
        # Expand segments to (segment_idx, cell_id) pairs using vectorized operations
        # Step 1: Compute number of cells each segment spans
        cell_ranges = cell_max - cell_min + 1  # (N, 2)
        num_cells_per_seg = cell_ranges[:, 0] * cell_ranges[:, 1]  # (N,)

        # Zero out segments that don't intersect grid (prevents false positives)
        num_cells_per_seg = torch.where(
            intersects_grid,
            num_cells_per_seg,
            torch.tensor(0, dtype=torch.long, device=self.device)
        )
        total_pairs = num_cells_per_seg.sum().item()

        if total_pairs == 0:
            num_cells = grid_dims[0].item() * grid_dims[1].item()
            if profile is not None:
                profile['vectorized_expansion'] = time.time() - t0
                profile['num_pairs'] = 0
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.zeros(num_cells + 1, dtype=torch.long, device=self.device),
            )

        # Step 2: Create segment indices repeated by number of cells
        segment_indices = torch.arange(len(self.segment_starts), device=self.device)
        segment_indices_expanded = segment_indices.repeat_interleave(num_cells_per_seg)

        # Step 3: Generate local cell indices for each segment
        # Compute cumulative offsets to map flat indices to segments
        offsets = torch.cat([
            torch.tensor([0], device=self.device, dtype=torch.long),
            num_cells_per_seg.cumsum(0)
        ])

        # For each pair, compute its local index within the segment's cell range
        flat_indices = torch.arange(total_pairs, device=self.device, dtype=torch.long)
        local_indices = flat_indices - offsets[segment_indices_expanded]

        # Step 4: Unravel local indices to 2D offsets
        # local_idx = cy_offset * width + cx_offset
        widths = cell_ranges[:, 0]  # x-dimension width
        widths_expanded = widths[segment_indices_expanded]

        cx_offset = local_indices % widths_expanded
        cy_offset = local_indices // widths_expanded

        # Step 5: Convert offsets to absolute cell coordinates
        cell_min_expanded = cell_min[segment_indices_expanded]  # (total_pairs, 2)
        cx = cell_min_expanded[:, 0] + cx_offset
        cy = cell_min_expanded[:, 1] + cy_offset

        # Convert to linear cell IDs
        grid_width = grid_dims[0]
        cell_ids = cy * grid_width + cx

        if profile is not None:
            profile['vectorized_expansion'] = time.time() - t0
            profile['num_pairs'] = total_pairs

        t0 = time.time()
        # Sort by cell_id
        sorted_order = torch.argsort(cell_ids)
        sorted_segment_indices = segment_indices_expanded[sorted_order]
        sorted_cell_ids = cell_ids[sorted_order]
        if profile is not None:
            profile['sorting'] = time.time() - t0

        t0 = time.time()
        # Build CSR offsets using bincount
        num_cells = grid_dims[0].item() * grid_dims[1].item()
        cell_counts = torch.bincount(sorted_cell_ids, minlength=num_cells)
        cell_offsets = torch.cat(
            [
                torch.tensor([0], dtype=torch.long, device=self.device),
                torch.cumsum(cell_counts, dim=0),
            ]
        )
        if profile is not None:
            profile['csr_build'] = time.time() - t0

        return sorted_segment_indices, cell_offsets

    def _index_points(
        self,
        grid_origin: torch.Tensor,
        cell_size: float,
        grid_dims: torch.Tensor,
        expansion_distance: float,
        profile: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build CSR index for points.

        Points are treated as having a square bbox of size 2*expansion_distance.

        Args:
            profile: Optional dict to store timing information

        Returns:
            Tuple of (cell_point_indices, cell_point_offsets) in CSR format
        """
        import time
        if self.point_coords.shape[0] == 0:
            num_cells = grid_dims[0].item() * grid_dims[1].item()
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.zeros(num_cells + 1, dtype=torch.long, device=self.device),
            )

        t0 = time.time()
        # Compute expanded bbox for each point
        point_bbox_min = self.point_coords - expansion_distance
        point_bbox_max = self.point_coords + expansion_distance
        if profile is not None:
            profile['bbox_compute'] = time.time() - t0

        t0 = time.time()
        # Convert to cell coordinates
        cell_min_unclamped = torch.floor((point_bbox_min - grid_origin) / cell_size).long()
        cell_max_unclamped = torch.floor((point_bbox_max - grid_origin) / cell_size).long()

        # Clamp to grid bounds
        zeros = torch.zeros_like(grid_dims)
        grid_max = grid_dims - 1
        cell_min = torch.maximum(cell_min_unclamped, zeros)
        cell_min = torch.minimum(cell_min, grid_max)
        cell_max = torch.maximum(cell_max_unclamped, zeros)
        cell_max = torch.minimum(cell_max, grid_max)

        # Check if point bbox intersects grid [0, grid_dims-1]
        # Point intersects if: cell_max_unclamped >= 0 AND cell_min_unclamped < grid_dims
        intersects_grid = (
            (cell_max_unclamped[:, 0] >= 0) &
            (cell_max_unclamped[:, 1] >= 0) &
            (cell_min_unclamped[:, 0] < grid_dims[0]) &
            (cell_min_unclamped[:, 1] < grid_dims[1])
        )
        if profile is not None:
            profile['cell_coords'] = time.time() - t0

        t0 = time.time()
        # Expand points to (point_idx, cell_id) pairs using vectorized operations
        # Step 1: Compute number of cells each point spans
        cell_ranges = cell_max - cell_min + 1  # (N, 2)
        num_cells_per_point = cell_ranges[:, 0] * cell_ranges[:, 1]  # (N,)

        # Zero out points that don't intersect grid (prevents false positives)
        num_cells_per_point = torch.where(
            intersects_grid,
            num_cells_per_point,
            torch.tensor(0, dtype=torch.long, device=self.device)
        )
        total_pairs = num_cells_per_point.sum().item()

        if total_pairs == 0:
            num_cells = grid_dims[0].item() * grid_dims[1].item()
            if profile is not None:
                profile['vectorized_expansion'] = time.time() - t0
                profile['num_pairs'] = 0
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.zeros(num_cells + 1, dtype=torch.long, device=self.device),
            )

        # Step 2: Create point indices repeated by number of cells
        point_indices = torch.arange(len(self.point_coords), device=self.device)
        point_indices_expanded = point_indices.repeat_interleave(num_cells_per_point)

        # Step 3: Generate local cell indices for each point
        # Compute cumulative offsets to map flat indices to points
        offsets = torch.cat([
            torch.tensor([0], device=self.device, dtype=torch.long),
            num_cells_per_point.cumsum(0)
        ])

        # For each pair, compute its local index within the point's cell range
        flat_indices = torch.arange(total_pairs, device=self.device, dtype=torch.long)
        local_indices = flat_indices - offsets[point_indices_expanded]

        # Step 4: Unravel local indices to 2D offsets
        # local_idx = cy_offset * width + cx_offset
        widths = cell_ranges[:, 0]  # x-dimension width
        widths_expanded = widths[point_indices_expanded]

        cx_offset = local_indices % widths_expanded
        cy_offset = local_indices // widths_expanded

        # Step 5: Convert offsets to absolute cell coordinates
        cell_min_expanded = cell_min[point_indices_expanded]  # (total_pairs, 2)
        cx = cell_min_expanded[:, 0] + cx_offset
        cy = cell_min_expanded[:, 1] + cy_offset

        # Convert to linear cell IDs
        grid_width = grid_dims[0]
        cell_ids = cy * grid_width + cx

        if profile is not None:
            profile['vectorized_expansion'] = time.time() - t0
            profile['num_pairs'] = total_pairs

        t0 = time.time()
        # Sort by cell_id
        sorted_order = torch.argsort(cell_ids)
        sorted_point_indices = point_indices_expanded[sorted_order]
        sorted_cell_ids = cell_ids[sorted_order]
        if profile is not None:
            profile['sorting'] = time.time() - t0

        t0 = time.time()
        # Build CSR offsets using bincount
        num_cells = grid_dims[0].item() * grid_dims[1].item()
        cell_counts = torch.bincount(sorted_cell_ids, minlength=num_cells)
        cell_offsets = torch.cat(
            [
                torch.tensor([0], dtype=torch.long, device=self.device),
                torch.cumsum(cell_counts, dim=0),
            ]
        )
        if profile is not None:
            profile['csr_build'] = time.time() - t0

        return sorted_point_indices, cell_offsets


def _process_polygon(
    polygon: shapely.Polygon,
    geom_idx: int,
    segment_starts_list: list,
    segment_ends_list: list,
    segment_to_geom_list: list,
    polygon_vertices_list: list,
    polygon_ranges_list: list,
    polygon_geom_indices_list: list,
    vertex_offset: int,
):
    """Process a single polygon, adding its data to the collection lists."""
    # Get exterior ring coordinates (remove duplicate closing point)
    ring_coords = list(polygon.exterior.coords)[:-1]

    # Add segments
    for i in range(len(ring_coords)):
        segment_starts_list.append(ring_coords[i])
        segment_ends_list.append(ring_coords[(i + 1) % len(ring_coords)])
        segment_to_geom_list.append(geom_idx)

    # Add polygon vertices for inside/outside test
    polygon_vertices_list.extend(ring_coords)
    polygon_ranges_list.append([vertex_offset, vertex_offset + len(ring_coords)])
    polygon_geom_indices_list.append(geom_idx)
