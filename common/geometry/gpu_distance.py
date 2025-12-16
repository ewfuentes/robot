"""GPU-accelerated 2D geometry distance computations.

This module provides batched distance computations between points and various
2D geometric primitives (points, line segments, polygons) using PyTorch for
GPU acceleration.
"""

import common.torch.load_torch_deps  # noqa: F401  # Must be imported before torch
import torch


def point_to_point_distance(
    query_points: torch.Tensor,
    target_points: torch.Tensor,
) -> torch.Tensor:
    """Compute Euclidean distance between point pairs.

    Args:
        query_points: (..., 2) tensor of query point coordinates
        target_points: (..., 2) tensor of target point coordinates

    Returns:
        (...,) tensor of distances. Supports broadcasting.
    """
    return torch.linalg.vector_norm(query_points - target_points, dim=-1)


def point_to_segment_distance(
    query_points: torch.Tensor,
    segment_starts: torch.Tensor,
    segment_ends: torch.Tensor,
) -> torch.Tensor:
    """Compute distance from each query point to each line segment.

    Uses parametric projection: the closest point on segment AB to point P
    is A + t*(B-A) where t = clamp(dot(AP, AB) / |AB|^2, 0, 1).

    Args:
        query_points: (Q, 2) tensor of query point coordinates
        segment_starts: (S, 2) tensor of segment start coordinates
        segment_ends: (S, 2) tensor of segment end coordinates

    Returns:
        (Q, S) tensor where result[i, j] is distance from query_points[i]
        to the line segment from segment_starts[j] to segment_ends[j].
    """
    # AB = B - A, shape (S, 2)
    AB = segment_ends - segment_starts

    # |AB|^2, shape (S,)
    AB_norm_sq = (AB * AB).sum(dim=-1)

    # Handle degenerate segments (start == end) by treating as points
    # Add small epsilon to avoid division by zero
    AB_norm_sq_safe = AB_norm_sq.clamp(min=1e-12)

    # AP = P - A for each query point and segment
    # query_points: (Q, 2), segment_starts: (S, 2)
    # AP: (Q, S, 2)
    AP = query_points.unsqueeze(1) - segment_starts.unsqueeze(0)

    # dot(AP, AB) for each pair, shape (Q, S)
    # AB needs to be (1, S, 2) for broadcasting
    AP_dot_AB = (AP * AB.unsqueeze(0)).sum(dim=-1)

    # t parameter, clamped to [0, 1], shape (Q, S)
    t = (AP_dot_AB / AB_norm_sq_safe.unsqueeze(0)).clamp(0, 1)

    # Closest point on segment: A + t * AB, shape (Q, S, 2)
    closest = segment_starts.unsqueeze(0) + t.unsqueeze(-1) * AB.unsqueeze(0)

    # Distance from query point to closest point, shape (Q, S)
    distances = torch.linalg.vector_norm(query_points.unsqueeze(1) - closest, dim=-1)

    return distances


def point_to_segments_min_distance(
    query_points: torch.Tensor,
    segment_starts: torch.Tensor,
    segment_ends: torch.Tensor,
    segment_to_group: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """Compute minimum distance from each point to each group of segments.

    This is useful for computing distance to polylines (groups of connected
    segments) or polygon boundaries.

    Args:
        query_points: (Q, 2) tensor of query point coordinates
        segment_starts: (S, 2) tensor of segment start coordinates
        segment_ends: (S, 2) tensor of segment end coordinates
        segment_to_group: (S,) tensor mapping each segment to its group index
        num_groups: Number of groups

    Returns:
        (Q, num_groups) tensor where result[i, j] is the minimum distance
        from query_points[i] to any segment in group j.
    """
    # Compute all pairwise distances, shape (Q, S)
    all_distances = point_to_segment_distance(query_points, segment_starts, segment_ends)

    Q = query_points.shape[0]
    device = query_points.device

    # Initialize with infinity
    min_distances = torch.full(
        (Q, num_groups), float("inf"), device=device, dtype=all_distances.dtype
    )

    # Expand segment_to_group to match all_distances shape for scatter
    # segment_to_group: (S,) -> (Q, S)
    group_indices = segment_to_group.unsqueeze(0).expand(Q, -1)

    # Scatter minimum distances into groups
    min_distances.scatter_reduce_(1, group_indices, all_distances, reduce="amin")

    return min_distances


def point_in_polygon_winding(
    query_points: torch.Tensor,
    polygon_vertices: torch.Tensor,
    polygon_ranges: torch.Tensor,
) -> torch.Tensor:
    """Test if points are inside polygons using winding number algorithm.

    The winding number counts how many times the polygon winds around the point.
    A non-zero winding number means the point is inside.

    Args:
        query_points: (Q, 2) tensor of query point coordinates
        polygon_vertices: (V, 2) tensor of all polygon vertices concatenated
        polygon_ranges: (P, 2) tensor where each row is [start, end) indices
            into polygon_vertices for that polygon's vertices

    Returns:
        (Q, P) boolean tensor where result[i, j] is True if query_points[i]
        is inside polygon j.

    Note:
        Polygons are assumed to be implicitly closed - the first and last vertices
        are connected automatically. Do NOT include a duplicate closing vertex.
    """
    Q = query_points.shape[0]
    P = polygon_ranges.shape[0]
    device = query_points.device

    # Initialize winding numbers
    winding = torch.zeros((Q, P), device=device, dtype=torch.float32)

    # Process each polygon separately (variable vertex counts make batching complex)
    for poly_idx in range(P):
        start_idx = polygon_ranges[poly_idx, 0].item()
        end_idx = polygon_ranges[poly_idx, 1].item()

        assert end_idx > start_idx, f"Polygon {poly_idx} has no vertices"

        # Get polygon vertices for this polygon
        poly_verts = polygon_vertices[start_idx:end_idx]  # (N, 2)
        N = poly_verts.shape[0]

        # Create edges: from vertex i to vertex (i+1) % N
        v1 = poly_verts  # (N, 2)
        v2 = torch.roll(poly_verts, -1, dims=0)  # (N, 2)

        # Translate vertices relative to each query point
        # query_points: (Q, 2), v1: (N, 2)
        # v1_rel: (Q, N, 2)
        v1_rel = v1.unsqueeze(0) - query_points.unsqueeze(1)
        v2_rel = v2.unsqueeze(0) - query_points.unsqueeze(1)

        # Compute cross product (signed area) for each edge relative to each point
        # cross = v1.x * v2.y - v1.y * v2.x
        cross = v1_rel[..., 0] * v2_rel[..., 1] - v1_rel[..., 1] * v2_rel[..., 0]

        # Check if edge crosses the positive x-axis from the query point
        # An upward crossing (v1.y <= 0 < v2.y) with point to the left (cross > 0)
        # contributes +1 to the winding number
        # A downward crossing (v2.y <= 0 < v1.y) with point to the right (cross < 0)
        # contributes -1 to the winding number

        y1 = v1_rel[..., 1]  # (Q, N)
        y2 = v2_rel[..., 1]  # (Q, N)

        # Upward crossing: y1 <= 0 and y2 > 0 and cross > 0
        upward = (y1 <= 0) & (y2 > 0) & (cross > 0)

        # Downward crossing: y1 > 0 and y2 <= 0 and cross < 0
        downward = (y1 > 0) & (y2 <= 0) & (cross < 0)

        # Winding contribution: +1 for upward, -1 for downward
        winding_contrib = upward.float() - downward.float()

        # Sum over edges for this polygon
        winding[:, poly_idx] = winding_contrib.sum(dim=1)

    # Point is inside if winding number is non-zero
    return winding != 0


def signed_distance_to_polygons(
    query_points: torch.Tensor,
    polygon_vertices: torch.Tensor,
    polygon_ranges: torch.Tensor,
) -> torch.Tensor:
    """Compute signed distance from points to polygons.

    The signed distance is negative if the point is inside the polygon,
    positive if outside. The magnitude is the distance to the nearest
    edge of the polygon.

    Args:
        query_points: (Q, 2) tensor of query point coordinates
        polygon_vertices: (V, 2) tensor of all polygon vertices concatenated
        polygon_ranges: (P, 2) tensor where each row is [start, end) indices
            into polygon_vertices for that polygon's vertices

    Returns:
        (Q, P) tensor of signed distances where result[i, j] is the signed
        distance from query_points[i] to polygon j. Negative means inside.
    """
    Q = query_points.shape[0]
    P = polygon_ranges.shape[0]
    device = query_points.device

    # Build segment representation of polygon boundaries
    segment_starts_list = []
    segment_ends_list = []
    segment_to_polygon = []

    for poly_idx in range(P):
        start_idx = polygon_ranges[poly_idx, 0].item()
        end_idx = polygon_ranges[poly_idx, 1].item()

        if end_idx <= start_idx:
            continue

        poly_verts = polygon_vertices[start_idx:end_idx]
        N = poly_verts.shape[0]

        # Create edges
        segment_starts_list.append(poly_verts)
        segment_ends_list.append(torch.roll(poly_verts, -1, dims=0))
        segment_to_polygon.extend([poly_idx] * N)

    if not segment_starts_list:
        return torch.zeros((Q, P), device=device)

    segment_starts = torch.cat(segment_starts_list, dim=0)
    segment_ends = torch.cat(segment_ends_list, dim=0)
    segment_to_polygon_tensor = torch.tensor(
        segment_to_polygon, device=device, dtype=torch.long
    )

    # Compute minimum distance to boundary for each polygon
    boundary_distances = point_to_segments_min_distance(
        query_points, segment_starts, segment_ends, segment_to_polygon_tensor, P
    )

    # Determine inside/outside
    inside = point_in_polygon_winding(query_points, polygon_vertices, polygon_ranges)

    # Signed distance: negative inside, positive outside
    signed_distances = torch.where(inside, -boundary_distances, boundary_distances)

    return signed_distances
