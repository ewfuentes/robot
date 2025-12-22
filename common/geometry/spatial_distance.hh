#pragma once

#include <tuple>
#include "torch/torch.h"

namespace robot::geometry {

/**
 * Query distances from points to geometries using GPU-accelerated spatial index.
 *
 * This function uses a CUDA kernel with spatial indexing to efficiently compute
 * distances from query points to geometries. It processes particles sorted by
 * cell ID, with one CUDA block per cell loading segments into shared memory.
 *
 * @param query_points (N, 2) float32 tensor - query point coordinates
 * @param segment_starts (M_seg, 2) float32 tensor - segment start coordinates
 * @param segment_ends (M_seg, 2) float32 tensor - segment end coordinates
 * @param segment_to_geom (M_seg,) int64 tensor - segment to geometry mapping
 * @param point_coords (M_pt, 2) float32 tensor - point coordinates
 * @param point_to_geom (M_pt,) int64 tensor - point to geometry mapping
 * @param geometry_types (G,) int64 tensor - geometry type for each geometry
 * @param polygon_vertices (V, 2) float32 tensor - polygon vertices
 * @param polygon_ranges (P, 2) int64 tensor - polygon vertex ranges
 * @param polygon_geom_indices (P,) int64 tensor - polygon to geometry mapping
 * @param num_geometries Total number of geometries
 * @param cell_segment_indices CSR: segment IDs per cell (sorted by geom_id within cell)
 * @param cell_offsets CSR: segment cell ranges
 * @param cell_geom_indices CSR: unique geometry IDs per cell
 * @param cell_geom_offsets CSR: geometry cell ranges
 * @param cell_point_indices CSR: point IDs per cell (sorted by geom_id within cell)
 * @param cell_point_offsets CSR: point cell ranges
 * @param cell_point_geom_indices CSR: unique geometry IDs per cell for points
 * @param cell_point_geom_offsets CSR: geometry cell ranges for points
 * @param grid_origin (2,) float32 tensor - grid min corner
 * @param cell_size Grid cell size (float)
 * @param grid_dims (2,) int64 tensor - (nx, ny) cells
 *
 * @return tuple of (particle_indices, geometry_indices, distances)
 *         - particle_indices: (K,) int64 tensor - query point indices
 *         - geometry_indices: (K,) int64 tensor - geometry indices
 *         - distances: (K,) float32 tensor - distances
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> query_distances_cuda(
    const torch::Tensor& query_points, const torch::Tensor& segment_starts,
    const torch::Tensor& segment_ends, const torch::Tensor& segment_to_geom,
    const torch::Tensor& point_coords, const torch::Tensor& point_to_geom,
    const torch::Tensor& geometry_types, const torch::Tensor& polygon_vertices,
    const torch::Tensor& polygon_ranges, const torch::Tensor& polygon_geom_indices,
    int64_t num_geometries, const torch::Tensor& cell_segment_indices,
    const torch::Tensor& cell_offsets, const torch::Tensor& cell_geom_indices,
    const torch::Tensor& cell_geom_offsets, const torch::Tensor& cell_point_indices,
    const torch::Tensor& cell_point_offsets, const torch::Tensor& cell_point_geom_indices,
    const torch::Tensor& cell_point_geom_offsets, const torch::Tensor& grid_origin, float cell_size,
    const torch::Tensor& grid_dims);

/**
 * Test if points are inside polygon geometries using winding number algorithm.
 *
 * Uses CUDA kernel with one block per polygon geometry. Handles polygons with
 * holes correctly by summing winding numbers across all rings of a geometry.
 *
 * Ring vertices are read from segment_starts to avoid duplicate storage.
 *
 * @param query_points (N, 2) float32 tensor - query point coordinates
 * @param segment_starts (S, 2) float32 tensor - segment start coordinates
 *        (ring vertices are consecutive segment_starts for each ring)
 * @param polygon_segment_ranges (R, 2) int64 tensor - [start, end) segment
 *        indices per ring (vertices are segment_starts[start:end])
 * @param polygon_geom_indices (R,) int64 tensor - geometry index per ring
 * @param geom_ring_offsets (G_poly+1,) int64 tensor - CSR offsets mapping
 *        geometry index to its rings in polygon_segment_ranges
 *
 * @return (N, G_poly) bool tensor where result[i, j] is true if query_points[i]
 *         is inside polygon geometry j
 */
torch::Tensor point_in_polygon_cuda(
    const torch::Tensor& query_points,
    const torch::Tensor& segment_starts,
    const torch::Tensor& polygon_segment_ranges,
    const torch::Tensor& polygon_geom_indices,
    const torch::Tensor& geom_ring_offsets);

/**
 * Sparse point-in-polygon test using winding number algorithm.
 *
 * Takes explicit candidate (point, polygon) pairs instead of dense matrix.
 * Uses spatial indexing to only check points against nearby polygons.
 *
 * @param query_points (N, 2) float32 tensor - query point coordinates
 * @param candidate_point_idx (K,) int64 tensor - query point indices to check
 * @param candidate_poly_idx (K,) int64 tensor - polygon geometry indices to check
 * @param segment_starts (S, 2) float32 tensor - ring vertices
 * @param polygon_segment_ranges (R, 2) int64 tensor - [start, end) segment indices per ring
 * @param geom_ring_offsets (G_poly+1,) int64 tensor - CSR offsets for rings per polygon
 *
 * @return (K,) bool tensor where result[i] is true if point candidate_point_idx[i]
 *         is inside polygon candidate_poly_idx[i]
 */
torch::Tensor point_in_polygon_sparse_cuda(
    const torch::Tensor& query_points,
    const torch::Tensor& candidate_point_idx,
    const torch::Tensor& candidate_poly_idx,
    const torch::Tensor& segment_starts,
    const torch::Tensor& polygon_segment_ranges,
    const torch::Tensor& geom_ring_offsets);

}  // namespace robot::geometry
