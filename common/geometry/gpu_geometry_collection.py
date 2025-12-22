"""GPU-accelerated geometry collection and spatial index.

This module provides high-level abstractions for working with collections
of 2D geometries (Points, LineStrings, Polygons, MultiPolygons) on GPU.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence
import common.geometry.spatial_distance_python as sdp

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
        cell_segment_indices: (K,) tensor - segment indices sorted by (cell, geom_id)
        cell_offsets: (num_cells+1,) tensor - CSR row pointers for segments
        cell_point_indices: (P',) tensor - point indices sorted by (cell, geom_id)
        cell_point_offsets: (num_cells+1,) tensor - CSR row pointers for points
        cell_geom_indices: (G',) tensor - unique geometry IDs per cell (CSR format)
        cell_geom_offsets: (num_cells+1,) tensor - CSR row pointers for geometries
        cell_point_geom_indices: (G'',) tensor - unique geometry IDs per cell for points
        cell_point_geom_offsets: (num_cells+1,) tensor - CSR row pointers for point geometries
        cell_polygon_indices: (K',) tensor - polygon geometry indices per cell (CSR format)
        cell_polygon_offsets: (num_cells+1,) tensor - CSR row pointers for polygons
    """

    grid_origin: torch.Tensor
    cell_size: float
    expansion_distance: float
    grid_dims: torch.Tensor
    cell_segment_indices: torch.Tensor
    cell_offsets: torch.Tensor
    cell_point_indices: torch.Tensor
    cell_point_offsets: torch.Tensor
    cell_geom_indices: torch.Tensor
    cell_geom_offsets: torch.Tensor
    cell_point_geom_indices: torch.Tensor
    cell_point_geom_offsets: torch.Tensor
    cell_polygon_indices: torch.Tensor
    cell_polygon_offsets: torch.Tensor


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
        polygon_segment_ranges: (R, 2) tensor of [start, end) segment indices per ring.
            Ring vertices are segment_starts[start:end].
        polygon_geom_indices: (R,) tensor mapping each ring to geometry index.
        geom_ring_offsets: (G_poly+1,) CSR offsets for rings per polygon geometry.
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
    # polygon_segment_ranges: (R, 2) tensor of [start, end) segment indices per ring
    # Ring vertices can be read from segment_starts[start:end]
    polygon_segment_ranges: torch.Tensor
    polygon_geom_indices: torch.Tensor
    # CSR offsets mapping polygon geometry index to its rings in polygon_segment_ranges
    # geom_ring_offsets[i:i+1] gives the range of rings for polygon geometry i
    geom_ring_offsets: torch.Tensor

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
        polygon_segment_ranges_list = []
        polygon_geom_indices_list = []

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
                    polygon_segment_ranges_list,
                    polygon_geom_indices_list,
                )

            elif geom_type == "MultiPolygon":
                geometry_types.append(GeometryType.MULTIPOLYGON)
                for poly in geom.geoms:
                    _process_polygon(
                        poly,
                        geom_idx,
                        segment_starts_list,
                        segment_ends_list,
                        segment_to_geom_list,
                        polygon_segment_ranges_list,
                        polygon_geom_indices_list,
                    )

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
        if polygon_segment_ranges_list:
            polygon_segment_ranges = torch.tensor(
                polygon_segment_ranges_list, dtype=torch.long, device=device
            )
            polygon_geom_indices = torch.tensor(
                polygon_geom_indices_list, dtype=torch.long, device=device
            )

            # Compute geom_ring_offsets: CSR offsets mapping unique polygon geometry
            # indices to their rings. Rings are already ordered by geometry index
            # from the way _process_polygon is called.
            unique_geoms, counts = torch.unique_consecutive(
                polygon_geom_indices, return_counts=True
            )
            geom_ring_offsets = torch.cat([
                torch.zeros(1, dtype=torch.long, device=device),
                counts.cumsum(0)
            ])
        else:
            polygon_segment_ranges = torch.empty((0, 2), dtype=torch.long, device=device)
            polygon_geom_indices = torch.empty((0,), dtype=torch.long, device=device)
            geom_ring_offsets = torch.zeros(1, dtype=torch.long, device=device)

        return cls(
            device=device,
            num_geometries=num_geometries,
            geometry_types=geometry_types_tensor,
            segment_starts=segment_starts,
            segment_ends=segment_ends,
            segment_to_geom=segment_to_geom,
            point_coords=point_coords,
            point_to_geom=point_to_geom,
            polygon_segment_ranges=polygon_segment_ranges,
            polygon_geom_indices=polygon_geom_indices,
            geom_ring_offsets=geom_ring_offsets,
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
        if self.polygon_segment_ranges.shape[0] > 0:
            # Reconstruct polygon_vertices and polygon_ranges for the fallback function
            # Ring vertices are segment_starts[seg_start:seg_end]
            polygon_vertices_list = []
            polygon_ranges_list = []
            vertex_offset = 0
            for i in range(self.polygon_segment_ranges.shape[0]):
                seg_start = self.polygon_segment_ranges[i, 0].item()
                seg_end = self.polygon_segment_ranges[i, 1].item()
                ring_verts = self.segment_starts[seg_start:seg_end]
                polygon_vertices_list.append(ring_verts)
                num_verts = seg_end - seg_start
                polygon_ranges_list.append([vertex_offset, vertex_offset + num_verts])
                vertex_offset += num_verts

            polygon_vertices = torch.cat(polygon_vertices_list, dim=0)
            polygon_ranges = torch.tensor(
                polygon_ranges_list, dtype=torch.long, device=self.device
            )

            signed_distances = gpu_distance.signed_distance_to_polygons(
                query_points, polygon_vertices, polygon_ranges
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
                    cell_geom_indices=torch.empty(0, dtype=torch.long, device=self.device),
                    cell_geom_offsets=torch.zeros(num_cells + 1, dtype=torch.long, device=self.device),
                    cell_point_geom_indices=torch.empty(0, dtype=torch.long, device=self.device),
                    cell_point_geom_offsets=torch.zeros(num_cells + 1, dtype=torch.long, device=self.device),
                    cell_polygon_indices=torch.empty(0, dtype=torch.long, device=self.device),
                    cell_polygon_offsets=torch.zeros(num_cells + 1, dtype=torch.long, device=self.device),
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
        cell_segment_indices, cell_offsets, cell_geom_indices, cell_geom_offsets = self._index_segments(
            grid_origin, cell_size, grid_dims, expansion_distance, seg_profile
        )
        if profile is not None:
            profile['segments'] = seg_profile

        # Step 3: Build point index
        pt_profile = {} if profile is not None else None
        cell_point_indices, cell_point_offsets, cell_point_geom_indices, cell_point_geom_offsets = self._index_points(
            grid_origin, cell_size, grid_dims, expansion_distance, pt_profile
        )
        if profile is not None:
            profile['points'] = pt_profile

        # Step 4: Build polygon bbox index
        cell_polygon_indices, cell_polygon_offsets = self._index_polygon_bboxes(
            grid_origin, cell_size, grid_dims, expansion_distance
        )

        # Step 5: Store index
        self.spatial_index = SpatialIndex(
            grid_origin=grid_origin,
            cell_size=cell_size,
            expansion_distance=expansion_distance,
            grid_dims=grid_dims,
            cell_segment_indices=cell_segment_indices,
            cell_offsets=cell_offsets,
            cell_point_indices=cell_point_indices,
            cell_point_offsets=cell_point_offsets,
            cell_geom_indices=cell_geom_indices,
            cell_geom_offsets=cell_geom_offsets,
            cell_point_geom_indices=cell_point_geom_indices,
            cell_point_geom_offsets=cell_point_geom_offsets,
            cell_polygon_indices=cell_polygon_indices,
            cell_polygon_offsets=cell_polygon_offsets,
        )

    def _index_segments(
        self,
        grid_origin: torch.Tensor,
        cell_size: float,
        grid_dims: torch.Tensor,
        expansion_distance: float,
        profile: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build CSR index for segments.

        Args:
            profile: Optional dict to store timing information

        Returns:
            Tuple of (cell_segment_indices, cell_offsets, cell_geom_indices, cell_geom_offsets)
            - cell_segment_indices: sorted by (cell_id, geom_id)
            - cell_offsets: CSR row pointers for segments per cell
            - cell_geom_indices: unique geometry IDs per cell (CSR format)
            - cell_geom_offsets: CSR row pointers for geometries per cell
        """
        import time

        if self.segment_starts.shape[0] == 0:
            num_cells = grid_dims[0].item() * grid_dims[1].item()
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.zeros(num_cells + 1, dtype=torch.long, device=self.device),
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
        # Get geometry IDs for each segment
        geom_ids_expanded = self.segment_to_geom[segment_indices_expanded]

        # Sort by (cell_id, geom_id) using combined key
        combined_key = cell_ids * self.num_geometries + geom_ids_expanded
        sorted_order = torch.argsort(combined_key)
        sorted_segment_indices = segment_indices_expanded[sorted_order]
        sorted_cell_ids = cell_ids[sorted_order]
        sorted_geom_ids = geom_ids_expanded[sorted_order]
        if profile is not None:
            profile['sorting'] = time.time() - t0

        t0 = time.time()
        # Build CSR offsets for segments per cell using bincount
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

        t0 = time.time()
        # Compute unique (cell_id, geom_id) pairs for CSR geometry index
        # Use combined key to find unique pairs
        combined_sorted = sorted_cell_ids * self.num_geometries + sorted_geom_ids
        unique_combined, unique_inverse = torch.unique(combined_sorted, return_inverse=True)
        unique_cell_ids = unique_combined // self.num_geometries
        unique_geom_ids = unique_combined % self.num_geometries

        # Build CSR offsets for geometries per cell
        # For each cell, count unique geometries
        cell_geom_counts = torch.bincount(unique_cell_ids, minlength=num_cells)
        cell_geom_offsets = torch.cat(
            [
                torch.tensor([0], dtype=torch.long, device=self.device),
                torch.cumsum(cell_geom_counts, dim=0),
            ]
        )
        cell_geom_indices = unique_geom_ids
        if profile is not None:
            profile['geom_csr_build'] = time.time() - t0

        return sorted_segment_indices, cell_offsets, cell_geom_indices, cell_geom_offsets

    def _index_points(
        self,
        grid_origin: torch.Tensor,
        cell_size: float,
        grid_dims: torch.Tensor,
        expansion_distance: float,
        profile: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build CSR index for points.

        Points are treated as having a square bbox of size 2*expansion_distance.

        Args:
            profile: Optional dict to store timing information

        Returns:
            Tuple of (cell_point_indices, cell_offsets, cell_geom_indices, cell_geom_offsets)
            - cell_point_indices: sorted by (cell_id, geom_id)
            - cell_offsets: CSR row pointers for points per cell
            - cell_geom_indices: unique geometry IDs per cell (CSR format)
            - cell_geom_offsets: CSR row pointers for geometries per cell
        """
        import time
        if self.point_coords.shape[0] == 0:
            num_cells = grid_dims[0].item() * grid_dims[1].item()
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.zeros(num_cells + 1, dtype=torch.long, device=self.device),
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
        # Get geometry IDs for each point
        geom_ids_expanded = self.point_to_geom[point_indices_expanded]

        # Sort by (cell_id, geom_id) using combined key
        combined_key = cell_ids * self.num_geometries + geom_ids_expanded
        sorted_order = torch.argsort(combined_key)
        sorted_point_indices = point_indices_expanded[sorted_order]
        sorted_cell_ids = cell_ids[sorted_order]
        sorted_geom_ids = geom_ids_expanded[sorted_order]
        if profile is not None:
            profile['sorting'] = time.time() - t0

        t0 = time.time()
        # Build CSR offsets for points per cell using bincount
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

        t0 = time.time()
        # Compute unique (cell_id, geom_id) pairs for CSR geometry index
        combined_sorted = sorted_cell_ids * self.num_geometries + sorted_geom_ids
        unique_combined, unique_inverse = torch.unique(combined_sorted, return_inverse=True)
        unique_cell_ids = unique_combined // self.num_geometries
        unique_geom_ids = unique_combined % self.num_geometries

        # Build CSR offsets for geometries per cell
        cell_geom_counts = torch.bincount(unique_cell_ids, minlength=num_cells)
        cell_geom_offsets = torch.cat(
            [
                torch.tensor([0], dtype=torch.long, device=self.device),
                torch.cumsum(cell_geom_counts, dim=0),
            ]
        )
        cell_geom_indices = unique_geom_ids
        if profile is not None:
            profile['geom_csr_build'] = time.time() - t0

        return sorted_point_indices, cell_offsets, cell_geom_indices, cell_geom_offsets

    def _index_polygon_bboxes(
        self,
        grid_origin: torch.Tensor,
        cell_size: float,
        grid_dims: torch.Tensor,
        expansion_distance: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build CSR index for polygon bounding boxes.

        Each polygon geometry gets one bounding box entry (union of all its rings).
        This enables sparse point-in-polygon queries by only checking polygons
        whose bboxes overlap with a query point's cell.

        Returns:
            Tuple of (cell_polygon_indices, cell_polygon_offsets)
            - cell_polygon_indices: polygon geometry indices per cell (CSR values)
            - cell_polygon_offsets: CSR row pointers for polygons per cell
        """
        num_cells = grid_dims[0].item() * grid_dims[1].item()
        num_polygon_geoms = self.geom_ring_offsets.size(0) - 1

        if num_polygon_geoms == 0:
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.zeros(num_cells + 1, dtype=torch.long, device=self.device),
            )

        # Get unique polygon geometry indices
        unique_poly_geoms = self.polygon_geom_indices.unique()
        num_unique_polys = len(unique_poly_geoms)

        # Create mapping from global geom index to local polygon index
        geom_to_poly_idx = torch.full((self.num_geometries,), -1,
                                      dtype=torch.long, device=self.device)
        geom_to_poly_idx[unique_poly_geoms] = torch.arange(num_unique_polys, device=self.device)

        # Vectorized bbox computation - no Python loops
        # Step 1: Get segment ranges for all rings
        seg_starts = self.polygon_segment_ranges[:, 0]  # (R,)
        seg_ends = self.polygon_segment_ranges[:, 1]    # (R,)
        ring_lengths = seg_ends - seg_starts            # (R,)

        # Step 2: Expand to get all vertex indices
        # For each ring, generate indices: seg_start, seg_start+1, ..., seg_end-1
        total_vertices = ring_lengths.sum()
        ring_indices = torch.arange(len(ring_lengths), device=self.device)
        ring_indices_expanded = ring_indices.repeat_interleave(ring_lengths)

        # Compute vertex indices within segment_starts
        ring_offsets = torch.cat([
            torch.zeros(1, dtype=torch.long, device=self.device),
            ring_lengths.cumsum(0)
        ])
        flat_vertex_idx = torch.arange(total_vertices, device=self.device, dtype=torch.long)
        local_vertex_idx = flat_vertex_idx - ring_offsets[ring_indices_expanded]
        seg_starts_expanded = seg_starts[ring_indices_expanded]
        vertex_indices = seg_starts_expanded + local_vertex_idx

        # Step 3: Gather all vertices
        all_vertices = self.segment_starts[vertex_indices]  # (total_vertices, 2)

        # Step 4: Map rings to polygon geometry indices
        poly_idx_per_ring = geom_to_poly_idx[self.polygon_geom_indices]  # (R,)
        poly_idx_per_vertex = poly_idx_per_ring[ring_indices_expanded]   # (total_vertices,)

        # Step 5: Compute bbox per polygon using scatter_reduce
        # Need to expand poly_idx for both x and y coordinates
        poly_idx_expanded = poly_idx_per_vertex.unsqueeze(1).expand(-1, 2)  # (total_vertices, 2)

        poly_bbox_min = torch.full((num_unique_polys, 2), float('inf'),
                                   dtype=torch.float32, device=self.device)
        poly_bbox_max = torch.full((num_unique_polys, 2), float('-inf'),
                                   dtype=torch.float32, device=self.device)

        poly_bbox_min.scatter_reduce_(0, poly_idx_expanded, all_vertices, reduce='amin')
        poly_bbox_max.scatter_reduce_(0, poly_idx_expanded, all_vertices, reduce='amax')

        # Expand bboxes by expansion_distance
        poly_bbox_min = poly_bbox_min - expansion_distance
        poly_bbox_max = poly_bbox_max + expansion_distance

        # Convert to cell coordinates
        cell_min_unclamped = torch.floor((poly_bbox_min - grid_origin) / cell_size).long()
        cell_max_unclamped = torch.floor((poly_bbox_max - grid_origin) / cell_size).long()

        # Clamp to grid bounds
        zeros = torch.zeros_like(grid_dims)
        grid_max = grid_dims - 1
        cell_min = torch.maximum(cell_min_unclamped, zeros)
        cell_min = torch.minimum(cell_min, grid_max)
        cell_max = torch.maximum(cell_max_unclamped, zeros)
        cell_max = torch.minimum(cell_max, grid_max)

        # Check if polygon bbox intersects grid
        intersects_grid = (
            (cell_max_unclamped[:, 0] >= 0) &
            (cell_max_unclamped[:, 1] >= 0) &
            (cell_min_unclamped[:, 0] < grid_dims[0]) &
            (cell_min_unclamped[:, 1] < grid_dims[1])
        )

        # Expand polygons to (polygon_idx, cell_id) pairs using vectorized operations
        cell_ranges = cell_max - cell_min + 1  # (P, 2)
        num_cells_per_poly = cell_ranges[:, 0] * cell_ranges[:, 1]  # (P,)

        # Zero out polygons that don't intersect grid
        num_cells_per_poly = torch.where(
            intersects_grid,
            num_cells_per_poly,
            torch.tensor(0, dtype=torch.long, device=self.device)
        )
        total_pairs = num_cells_per_poly.sum().item()

        if total_pairs == 0:
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.zeros(num_cells + 1, dtype=torch.long, device=self.device),
            )

        # Create polygon indices repeated by number of cells
        poly_indices = torch.arange(num_unique_polys, device=self.device)
        poly_indices_expanded = poly_indices.repeat_interleave(num_cells_per_poly)

        # Generate local cell indices for each polygon
        offsets = torch.cat([
            torch.tensor([0], device=self.device, dtype=torch.long),
            num_cells_per_poly.cumsum(0)
        ])

        flat_indices = torch.arange(total_pairs, device=self.device, dtype=torch.long)
        local_indices = flat_indices - offsets[poly_indices_expanded]

        # Unravel local indices to 2D offsets
        widths = cell_ranges[:, 0]
        widths_expanded = widths[poly_indices_expanded]

        cx_offset = local_indices % widths_expanded
        cy_offset = local_indices // widths_expanded

        # Convert to absolute cell coordinates
        cell_min_expanded = cell_min[poly_indices_expanded]
        cx = cell_min_expanded[:, 0] + cx_offset
        cy = cell_min_expanded[:, 1] + cy_offset

        # Convert to linear cell IDs
        grid_width = grid_dims[0]
        cell_ids = cy * grid_width + cx

        # Map local polygon indices back to global geometry indices
        global_geom_indices = unique_poly_geoms[poly_indices_expanded]

        # Sort by cell_id
        sorted_order = torch.argsort(cell_ids)
        sorted_cell_ids = cell_ids[sorted_order]
        sorted_geom_indices = global_geom_indices[sorted_order]

        # Remove duplicates (same polygon can appear multiple times per cell if
        # the bbox expansion spans it, but we only need one entry)
        combined_key = sorted_cell_ids * self.num_geometries + sorted_geom_indices
        unique_combined, unique_indices = torch.unique(combined_key, return_inverse=True)

        # Get first occurrence of each unique pair
        first_occurrence = torch.zeros(len(unique_combined), dtype=torch.long, device=self.device)
        first_occurrence.scatter_(0, unique_indices, torch.arange(len(unique_indices), device=self.device))

        unique_cell_ids = unique_combined // self.num_geometries
        unique_geom_ids = unique_combined % self.num_geometries

        # Build CSR offsets for polygons per cell
        cell_counts = torch.bincount(unique_cell_ids, minlength=num_cells)
        cell_polygon_offsets = torch.cat([
            torch.tensor([0], dtype=torch.long, device=self.device),
            torch.cumsum(cell_counts, dim=0),
        ])

        return unique_geom_ids, cell_polygon_offsets

    def query_distances_cuda(
        self,
        query_points: torch.Tensor,
    ) -> torch.Tensor:
        """Query distances using CUDA kernel (internal method).

        Args:
            query_points: (N, 2) tensor of query positions

        Returns:
            (K, 3) tensor: [particle_idx, geometry_idx, distance]
        """
        if self.spatial_index is None:
            raise ValueError(
                "Spatial index must be built before querying distances. "
                "Call build_spatial_index() first."
            )

        if query_points.shape[1] != 2:
            raise ValueError(f"query_points must have shape (N, 2), got {query_points.shape}")

        idx = self.spatial_index

        # Note: polygon_vertices, polygon_ranges, polygon_geom_indices are passed
        # as empty tensors because they're unused placeholder parameters in the
        # C++ function. Point-in-polygon is handled separately via point_in_polygon_cuda.
        empty_float = torch.empty((0, 2), dtype=torch.float32, device=self.device)
        empty_long = torch.empty((0, 2), dtype=torch.long, device=self.device)
        empty_long_1d = torch.empty((0,), dtype=torch.long, device=self.device)

        particle_indices, geometry_indices, distances = sdp.query_distances_cuda(
            query_points,
            self.segment_starts,
            self.segment_ends,
            self.segment_to_geom,
            self.point_coords,
            self.point_to_geom,
            self.geometry_types,
            empty_float,   # polygon_vertices (unused)
            empty_long,    # polygon_ranges (unused)
            empty_long_1d, # polygon_geom_indices (unused)
            self.num_geometries,
            idx.cell_segment_indices,
            idx.cell_offsets,
            idx.cell_geom_indices,
            idx.cell_geom_offsets,
            idx.cell_point_indices,
            idx.cell_point_offsets,
            idx.cell_point_geom_indices,
            idx.cell_point_geom_offsets,
            idx.grid_origin,
            idx.cell_size,
            idx.grid_dims,
        )

        # For polygon geometries, use sparse point-in-polygon check with spatial filtering
        # Only check points against polygons whose bboxes overlap with the point's cell
        num_polygon_geoms = self.geom_ring_offsets.size(0) - 1
        if num_polygon_geoms > 0 and idx.cell_polygon_indices.numel() > 0:
            # Get unique polygon geometry indices and create mapping
            unique_poly_geoms = self.polygon_geom_indices.unique()

            # Create a mapping from global geometry index to local polygon index
            geom_to_poly_idx = torch.full((self.num_geometries,), -1, dtype=torch.long, device=self.device)
            geom_to_poly_idx[unique_poly_geoms] = torch.arange(len(unique_poly_geoms), device=self.device)

            # Step 1: Hash query points to cells
            cell_coords = torch.floor((query_points - idx.grid_origin) / idx.cell_size).long()
            num_cells = idx.grid_dims[0] * idx.grid_dims[1]

            # Check which points are in bounds
            in_bounds = (
                (cell_coords[:, 0] >= 0) & (cell_coords[:, 0] < idx.grid_dims[0]) &
                (cell_coords[:, 1] >= 0) & (cell_coords[:, 1] < idx.grid_dims[1])
            )
            cell_ids = cell_coords[:, 1] * idx.grid_dims[0] + cell_coords[:, 0]
            cell_ids = torch.clamp(cell_ids, 0, num_cells - 1)

            # Step 2: Get candidate polygon counts per point using CSR lookup
            poly_starts = idx.cell_polygon_offsets[cell_ids]  # (N,)
            poly_ends = idx.cell_polygon_offsets[cell_ids + 1]  # (N,)
            num_candidates_per_point = poly_ends - poly_starts  # (N,)

            # Zero out candidates for out-of-bounds points
            num_candidates_per_point = torch.where(
                in_bounds,
                num_candidates_per_point,
                torch.tensor(0, dtype=torch.long, device=self.device)
            )
            total_candidates = num_candidates_per_point.sum().item()

            if total_candidates > 0:
                # Step 3: Build sparse (point_idx, polygon_idx) candidate pairs
                point_indices_local = torch.arange(len(query_points), device=self.device)
                candidate_point_idx = point_indices_local.repeat_interleave(num_candidates_per_point)

                # Generate polygon indices using CSR expansion
                offsets = torch.cat([
                    torch.tensor([0], device=self.device, dtype=torch.long),
                    num_candidates_per_point.cumsum(0)
                ])
                flat_indices = torch.arange(total_candidates, device=self.device, dtype=torch.long)
                local_poly_offset = flat_indices - offsets[candidate_point_idx]

                # Look up the actual polygon geometry indices
                poly_starts_expanded = poly_starts[candidate_point_idx]
                global_poly_indices = idx.cell_polygon_indices[poly_starts_expanded + local_poly_offset]

                # Convert global geometry indices to local polygon indices for geom_ring_offsets
                candidate_local_poly_idx = geom_to_poly_idx[global_poly_indices]

                # Step 4: Call sparse point-in-polygon kernel
                is_inside_sparse = sdp.point_in_polygon_sparse_cuda(
                    query_points,
                    candidate_point_idx,
                    candidate_local_poly_idx,
                    self.segment_starts,
                    self.polygon_segment_ranges,
                    self.geom_ring_offsets,
                )  # (K,) bool tensor

                # Step 5: For existing polygon results, set distance to 0 if inside
                # Use searchsorted for sparse key matching (avoids huge N*G lookup table)
                sparse_keys = candidate_point_idx * self.num_geometries + global_poly_indices

                if particle_indices.numel() > 0:
                    polygon_mask = (self.geometry_types[geometry_indices] == GeometryType.POLYGON) | \
                                  (self.geometry_types[geometry_indices] == GeometryType.MULTIPOLYGON)

                    # Process unconditionally - empty tensors are cheap
                    poly_particle_indices = particle_indices[polygon_mask]
                    poly_geom_indices = geometry_indices[polygon_mask]
                    existing_keys = poly_particle_indices * self.num_geometries + poly_geom_indices

                    if existing_keys.numel() > 0:
                        # Sort sparse keys for binary search
                        sparse_keys_sorted, sort_idx = sparse_keys.sort()
                        is_inside_sorted = is_inside_sparse[sort_idx]

                        # Find where existing_keys would be inserted in sorted sparse_keys
                        match_idx = torch.searchsorted(sparse_keys_sorted, existing_keys)

                        # Check if matches are valid (within bounds and keys match)
                        valid_match = match_idx < len(sparse_keys_sorted)
                        valid_match = valid_match & (sparse_keys_sorted[match_idx.clamp(max=len(sparse_keys_sorted)-1)] == existing_keys)

                        # Get is_inside for valid matches
                        is_inside = torch.zeros(len(existing_keys), dtype=torch.bool, device=self.device)
                        is_inside[valid_match] = is_inside_sorted[match_idx[valid_match]]

                        # Set distance to 0 for inside points using boolean indexing
                        # Create a full-size mask and combine with polygon_mask
                        inside_full_mask = torch.zeros(len(distances), dtype=torch.bool, device=self.device)
                        inside_full_mask[polygon_mask] = is_inside
                        distances[inside_full_mask] = 0.0

                # Step 6: Add entries for points inside polygons but not near edges
                # Filter using boolean indexing directly (no torch.where)
                inside_point_idx = candidate_point_idx[is_inside_sparse]
                inside_poly_geom_idx = global_poly_indices[is_inside_sparse]

                if inside_point_idx.numel() > 0:
                    # Build existing keys for deduplication
                    if particle_indices.numel() > 0:
                        polygon_mask = (self.geometry_types[geometry_indices] == GeometryType.POLYGON) | \
                                      (self.geometry_types[geometry_indices] == GeometryType.MULTIPOLYGON)
                        existing_keys = particle_indices[polygon_mask] * self.num_geometries + geometry_indices[polygon_mask]
                    else:
                        existing_keys = torch.empty(0, dtype=torch.long, device=self.device)

                    # Compute new keys
                    new_keys = inside_point_idx * self.num_geometries + inside_poly_geom_idx

                    # Find keys that don't exist in results yet using torch.isin (GPU-native)
                    new_mask = ~torch.isin(new_keys, existing_keys)

                    # Apply mask directly without .any() check - cat with empty is fine
                    new_p = inside_point_idx[new_mask]
                    new_g = inside_poly_geom_idx[new_mask]
                    new_d = torch.zeros(new_p.size(0), dtype=torch.float32, device=self.device)

                    particle_indices = torch.cat([particle_indices, new_p])
                    geometry_indices = torch.cat([geometry_indices, new_g])
                    distances = torch.cat([distances, new_d])

        # Stack into (K, 3) tensor for backwards compatibility
        if particle_indices.numel() == 0:
            return torch.empty((0, 3), dtype=torch.float32, device=self.device)

        return torch.stack([
            particle_indices.float(),
            geometry_indices.float(),
            distances,
        ], dim=1)


def _process_polygon(
    polygon: shapely.Polygon,
    geom_idx: int,
    segment_starts_list: list,
    segment_ends_list: list,
    segment_to_geom_list: list,
    polygon_segment_ranges_list: list,
    polygon_geom_indices_list: list,
) -> None:
    """Process a single polygon, adding its data to the collection lists.

    Processes the exterior ring and all interior rings (holes). All rings
    share the same geom_idx. The winding number algorithm handles holes
    correctly because exterior rings are CCW and hole rings are CW.

    Ring vertices are stored as segment_starts for consecutive segments,
    avoiding duplicate storage of coordinates.

    Args:
        polygon: Shapely polygon to process
        geom_idx: Geometry index for this polygon
        segment_starts_list: List to append segment start coordinates
        segment_ends_list: List to append segment end coordinates
        segment_to_geom_list: List to append geometry indices for segments
        polygon_segment_ranges_list: List to append [start, end) segment indices per ring
        polygon_geom_indices_list: List to append geometry indices for rings
    """
    # Ensure correct ring orientation: exterior CCW, holes CW
    # This is required for the winding number algorithm to work correctly
    polygon = shapely.geometry.polygon.orient(polygon)

    # Process exterior ring and all interior rings (holes)
    all_rings = [polygon.exterior] + list(polygon.interiors)

    for ring in all_rings:
        # Get ring coordinates (remove duplicate closing point)
        ring_coords = list(ring.coords)[:-1]

        if len(ring_coords) < 3:
            # Skip degenerate rings
            continue

        # Track segment range for this ring
        ring_segment_start = len(segment_starts_list)

        # Add segments for this ring
        for i in range(len(ring_coords)):
            segment_starts_list.append(ring_coords[i])
            segment_ends_list.append(ring_coords[(i + 1) % len(ring_coords)])
            segment_to_geom_list.append(geom_idx)

        ring_segment_end = len(segment_starts_list)

        # Store segment range for this ring (vertices are segment_starts[start:end])
        polygon_segment_ranges_list.append([ring_segment_start, ring_segment_end])
        polygon_geom_indices_list.append(geom_idx)
