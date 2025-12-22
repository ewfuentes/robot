#pragma once

#include <tuple>
#include "torch/torch.h"

namespace robot::geometry {

// ============================================================================
// Geometry Data Structures
// ============================================================================

/// Segment geometry data (LineStrings and Polygon boundaries)
struct SegmentGeometry {
    torch::Tensor starts;   // (S, 2) segment start coordinates
    torch::Tensor ends;     // (S, 2) segment end coordinates
    torch::Tensor to_geom;  // (S,) segment to geometry index mapping
};

/// Point geometry data
struct PointGeometry {
    torch::Tensor coords;   // (P, 2) point coordinates
    torch::Tensor to_geom;  // (P,) point to geometry index mapping
};

/// Polygon ring structure for point-in-polygon tests
struct PolygonRingData {
    torch::Tensor segment_ranges;    // (R, 2) [start, end) segment indices per ring
    torch::Tensor geom_indices;      // (R,) geometry index per ring
    torch::Tensor geom_ring_offsets; // (G_poly+1,) CSR offsets for rings per polygon
};

// ============================================================================
// Spatial Index Structures
// ============================================================================

/// Grid configuration for spatial hashing
struct GridConfig {
    torch::Tensor origin;  // (2,) grid min corner
    float cell_size;       // grid cell size
    torch::Tensor dims;    // (2,) cell counts (nx, ny)
};

/// Spatial index for segments (CSR format)
struct SegmentSpatialIndex {
    torch::Tensor segment_indices;  // indices into SegmentGeometry arrays
    torch::Tensor cell_offsets;     // (num_cells+1,) CSR row pointers
    torch::Tensor geom_indices;     // unique geometry IDs per cell
    torch::Tensor geom_offsets;     // (num_cells+1,) CSR row pointers for geometries
};

/// Spatial index for points (CSR format)
struct PointSpatialIndex {
    torch::Tensor point_indices;  // indices into PointGeometry arrays
    torch::Tensor cell_offsets;   // (num_cells+1,) CSR row pointers
    torch::Tensor geom_indices;   // unique geometry IDs per cell
    torch::Tensor geom_offsets;   // (num_cells+1,) CSR row pointers for geometries
};

/// Spatial index for polygon bounding boxes (CSR format)
struct PolygonSpatialIndex {
    torch::Tensor geom_indices;  // geometry IDs per cell
    torch::Tensor cell_offsets;  // (num_cells+1,) CSR row pointers
};

// ============================================================================
// Functions
// ============================================================================

/**
 * Query distances from points to geometries using GPU-accelerated spatial index.
 *
 * Uses a CUDA kernel with spatial indexing to efficiently compute distances
 * from query points to geometries. Processes particles sorted by cell ID,
 * with one CUDA block per cell loading segments into shared memory.
 *
 * For polygon geometries, also performs point-in-polygon checks and sets
 * distance to 0 for points inside polygons.
 *
 * @param query_points (N, 2) float32 tensor - query point coordinates
 * @param num_geometries Total number of geometries
 * @param segments Segment geometry data
 * @param points Point geometry data
 * @param poly_rings Polygon ring structure
 * @param grid Grid configuration
 * @param seg_idx Segment spatial index
 * @param pt_idx Point spatial index
 * @param poly_idx Polygon spatial index
 *
 * @return tuple of (particle_indices, geometry_indices, distances)
 *         - particle_indices: (K,) int64 tensor - query point indices
 *         - geometry_indices: (K,) int64 tensor - geometry indices
 *         - distances: (K,) float32 tensor - distances (0 if inside polygon)
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> query_distances_cuda(
    const torch::Tensor& query_points,
    int64_t num_geometries,
    const SegmentGeometry& segments,
    const PointGeometry& points,
    const PolygonRingData& poly_rings,
    const GridConfig& grid,
    const SegmentSpatialIndex& seg_idx,
    const PointSpatialIndex& pt_idx,
    const PolygonSpatialIndex& poly_idx);

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
