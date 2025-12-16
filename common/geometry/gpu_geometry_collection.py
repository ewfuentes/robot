"""GPU-accelerated geometry collection and spatial index.

This module provides high-level abstractions for working with collections
of 2D geometries (Points, LineStrings, Polygons, MultiPolygons) on GPU.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import numpy as np
import shapely
import torch
from torch_kdtree import build_kd_tree

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


def sample_geometry_boundary(
    geom: shapely.Geometry,
    spacing: float,
) -> np.ndarray:
    """Sample points along a geometry's boundary at regular intervals.

    Args:
        geom: A shapely geometry object.
        spacing: Target spacing between sample points.

    Returns:
        (N, 2) numpy array of sample point coordinates.
    """
    geom_type = geom.geom_type

    if geom_type == "Point":
        return np.array([[geom.x, geom.y]])

    elif geom_type == "LineString":
        return _sample_linestring(geom, spacing)

    elif geom_type == "Polygon":
        # Sample exterior ring
        return _sample_linestring(
            shapely.LinearRing(geom.exterior.coords), spacing
        )

    elif geom_type == "MultiPolygon":
        samples = []
        for poly in geom.geoms:
            samples.append(
                _sample_linestring(shapely.LinearRing(poly.exterior.coords), spacing)
            )
        return np.vstack(samples) if samples else np.empty((0, 2))

    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")


def _sample_linestring(geom: shapely.LineString, spacing: float) -> np.ndarray:
    """Sample points along a LineString at regular intervals."""
    length = geom.length
    if length == 0:
        # Degenerate case: return single point
        coords = list(geom.coords)
        return np.array([coords[0]])

    num_samples = max(2, int(np.ceil(length / spacing)) + 1)
    fractions = np.linspace(0, 1, num_samples)
    points = [geom.interpolate(f, normalized=True) for f in fractions]
    return np.array([[p.x, p.y] for p in points])


@dataclass
class GPUSpatialIndex:
    """GPU spatial index for fast nearest-neighbor queries.

    Uses torch_kdtree on sampled boundary points for candidate filtering,
    then computes exact distances for candidates.

    Attributes:
        collection: The underlying GPUGeometryCollection.
        kdtree: KD-tree built from sample points.
        sample_points: (N, 2) tensor of sampled boundary points.
        sample_to_geom: (N,) tensor mapping each sample to geometry index.
    """

    collection: GPUGeometryCollection
    kdtree: object  # torch_kdtree KDTree
    sample_points: torch.Tensor
    sample_to_geom: torch.Tensor

    @classmethod
    def build(
        cls,
        geometries: Sequence[shapely.Geometry],
        device: torch.device = torch.device("cpu"),
        sample_spacing: float = 50.0,
    ) -> "GPUSpatialIndex":
        """Build spatial index from shapely geometries.

        Args:
            geometries: Sequence of shapely geometry objects.
            device: Target torch device.
            sample_spacing: Target spacing between boundary sample points.

        Returns:
            GPUSpatialIndex ready for queries.
        """
        # Build the geometry collection
        collection = GPUGeometryCollection.from_shapely(geometries, device)

        # Sample boundary points from all geometries
        sample_points_list = []
        sample_to_geom_list = []

        for geom_idx, geom in enumerate(geometries):
            samples = sample_geometry_boundary(geom, sample_spacing)
            sample_points_list.append(samples)
            sample_to_geom_list.extend([geom_idx] * len(samples))

        sample_points = torch.tensor(
            np.vstack(sample_points_list), dtype=torch.float32, device=device
        )
        sample_to_geom = torch.tensor(
            sample_to_geom_list, dtype=torch.long, device=device
        )

        # Build KD-tree
        kdtree = build_kd_tree(sample_points)

        return cls(
            collection=collection,
            kdtree=kdtree,
            sample_points=sample_points,
            sample_to_geom=sample_to_geom,
        )

    def query_nearest(
        self,
        query_points: torch.Tensor,
        k: int = 10,
        max_distance: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find k nearest geometries to each query point.

        This is an approximate query using the KD-tree on sample points.
        For exact results, use query_within_distance.

        Args:
            query_points: (Q, 2) tensor of query point coordinates.
            k: Number of nearest geometries to find.
            max_distance: Optional maximum distance threshold.

        Returns:
            Tuple of:
                - (Q, K) tensor of geometry indices (-1 for invalid).
                - (Q, K) tensor of approximate distances.
        """
        Q = query_points.shape[0]
        device = query_points.device

        # Query KD-tree for nearest sample points
        k_samples = min(k * 3, self.sample_points.shape[0])  # Over-sample
        distances_sq, sample_idxs = self.kdtree.query(
            query_points, nr_nns_searches=k_samples
        )

        # Map sample indices to geometry indices
        geom_idxs = self.sample_to_geom[sample_idxs]  # (Q, k_samples)

        # For each query, find unique geometries and their best distances
        result_geom_idxs = torch.full((Q, k), -1, dtype=torch.long, device=device)
        result_distances = torch.full((Q, k), float("inf"), device=device)

        for q in range(Q):
            unique_geoms = geom_idxs[q].unique()
            if max_distance is not None:
                # Filter by approximate distance
                valid_mask = distances_sq[q] <= max_distance**2
                unique_geoms = geom_idxs[q][valid_mask].unique()

            # Get actual distances for unique geometries
            if len(unique_geoms) > 0:
                actual_distances = self.collection.distance_to_points(
                    query_points[q : q + 1]
                )[0, unique_geoms]

                # Sort and take top k
                sorted_idx = actual_distances.abs().argsort()
                num_results = min(k, len(unique_geoms))
                result_geom_idxs[q, :num_results] = unique_geoms[
                    sorted_idx[:num_results]
                ]
                result_distances[q, :num_results] = actual_distances[
                    sorted_idx[:num_results]
                ]

        return result_geom_idxs, result_distances

    def query_within_distance(
        self,
        query_points: torch.Tensor,
        distance: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Find all geometries within distance of each query point.

        Args:
            query_points: (Q, 2) tensor of query point coordinates.
            distance: Maximum distance threshold.

        Returns:
            Tuple of:
                - (M,) tensor of query point indices.
                - (M,) tensor of geometry indices.
                - (M,) tensor of exact distances (signed for polygons).
        """
        Q = query_points.shape[0]
        device = query_points.device

        # Query KD-tree for candidate geometries
        # Use larger radius to account for sampling
        search_radius = distance * 1.5
        k_samples = min(100, self.sample_points.shape[0])

        distances_sq, sample_idxs = self.kdtree.query(
            query_points, nr_nns_searches=k_samples
        )

        # Vectorized candidate collection
        # Filter by distance threshold
        valid_mask = distances_sq <= search_radius**2  # (Q, k_samples)

        # Get (query_idx, sample_idx) pairs for valid entries
        query_indices, sample_local_indices = torch.where(valid_mask)

        if query_indices.shape[0] == 0:
            return (
                torch.empty((0,), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.float32, device=device),
            )

        # Map sample indices to geometry indices
        sample_global_indices = sample_idxs[query_indices, sample_local_indices]
        geom_indices = self.sample_to_geom[sample_global_indices]

        # Get unique (query, geometry) pairs
        pair_keys = query_indices * self.collection.num_geometries + geom_indices
        unique_pair_keys, inverse_indices = torch.unique(pair_keys, return_inverse=True)

        # Extract unique query and geometry indices
        unique_query_idxs = unique_pair_keys // self.collection.num_geometries
        unique_geom_idxs = unique_pair_keys % self.collection.num_geometries

        # Compute exact distances for all unique query points
        queries_needing_distance = unique_query_idxs.unique()
        all_distances = self.collection.distance_to_points(
            query_points[queries_needing_distance]
        )

        # Map unique queries to row indices in all_distances
        query_to_row = torch.zeros(Q, dtype=torch.long, device=device)
        query_to_row[queries_needing_distance] = torch.arange(
            len(queries_needing_distance), device=device
        )

        # Extract distances for unique (query, geom) pairs
        distances = all_distances[
            query_to_row[unique_query_idxs], unique_geom_idxs
        ]

        # Filter by actual distance
        valid_mask = distances.abs() <= distance
        return (
            unique_query_idxs[valid_mask],
            unique_geom_idxs[valid_mask],
            distances[valid_mask],
        )
