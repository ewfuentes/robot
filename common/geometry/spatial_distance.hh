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

}  // namespace robot::geometry
