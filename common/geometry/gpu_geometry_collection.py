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
