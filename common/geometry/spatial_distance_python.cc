#include "common/geometry/spatial_distance.hh"
#include "torch/extension.h"

namespace robot::geometry {

// Wrapper that builds structs from individual tensor arguments
// This keeps the Python interface unchanged while using structs internally
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> query_distances_cuda_wrapper(
    const torch::Tensor& query_points,
    const torch::Tensor& segment_starts,
    const torch::Tensor& segment_ends,
    const torch::Tensor& segment_to_geom,
    const torch::Tensor& point_coords,
    const torch::Tensor& point_to_geom,
    int64_t num_geometries,
    const torch::Tensor& cell_segment_indices,
    const torch::Tensor& cell_offsets,
    const torch::Tensor& cell_geom_indices,
    const torch::Tensor& cell_geom_offsets,
    const torch::Tensor& cell_point_indices,
    const torch::Tensor& cell_point_offsets,
    const torch::Tensor& cell_point_geom_indices,
    const torch::Tensor& cell_point_geom_offsets,
    const torch::Tensor& grid_origin,
    float cell_size,
    const torch::Tensor& grid_dims,
    const torch::Tensor& cell_polygon_indices,
    const torch::Tensor& cell_polygon_offsets,
    const torch::Tensor& polygon_segment_ranges,
    const torch::Tensor& polygon_geom_indices,
    const torch::Tensor& geom_ring_offsets,
    bool debug = false) {

    // Build geometry structs
    SegmentGeometry segments{segment_starts, segment_ends, segment_to_geom};
    PointGeometry points{point_coords, point_to_geom};
    PolygonRingData poly_rings{polygon_segment_ranges, polygon_geom_indices, geom_ring_offsets};

    // Build spatial index structs
    GridConfig grid{grid_origin, cell_size, grid_dims};
    SegmentSpatialIndex seg_idx{cell_segment_indices, cell_offsets, cell_geom_indices, cell_geom_offsets};
    PointSpatialIndex pt_idx{cell_point_indices, cell_point_offsets, cell_point_geom_indices, cell_point_geom_offsets};
    PolygonSpatialIndex poly_idx{cell_polygon_indices, cell_polygon_offsets};

    // Call the actual function with structs
    return query_distances_cuda(query_points, num_geometries, segments, points, poly_rings,
                                grid, seg_idx, pt_idx, poly_idx, debug);
}

PYBIND11_MODULE(spatial_distance_python, m) {
    m.doc() = "GPU-accelerated spatial distance queries with CUDA";
    m.def("query_distances_cuda", &query_distances_cuda_wrapper,
          "Query distances from points to geometries using CUDA kernel with spatial index.\n"
          "Includes integrated point-in-polygon processing for polygon geometries.\n"
          "\n"
          "Args:\n"
          "    query_points: (N, 2) float32 tensor of query positions\n"
          "    segment_starts: (S, 2) float32 tensor of segment start coordinates\n"
          "    segment_ends: (S, 2) float32 tensor of segment end coordinates\n"
          "    segment_to_geom: (S,) int64 tensor mapping segments to geometry indices\n"
          "    point_coords: (P, 2) float32 tensor of point geometry coordinates\n"
          "    point_to_geom: (P,) int64 tensor mapping points to geometry indices\n"
          "    num_geometries: Total number of geometries\n"
          "    cell_segment_indices: CSR values for segments per cell\n"
          "    cell_offsets: CSR offsets for segments per cell\n"
          "    cell_geom_indices: CSR values for unique geometry IDs per cell\n"
          "    cell_geom_offsets: CSR offsets for geometries per cell\n"
          "    cell_point_indices: CSR values for points per cell\n"
          "    cell_point_offsets: CSR offsets for points per cell\n"
          "    cell_point_geom_indices: CSR values for unique point geometry IDs per cell\n"
          "    cell_point_geom_offsets: CSR offsets for point geometries per cell\n"
          "    grid_origin: (2,) float32 tensor of grid min corner\n"
          "    cell_size: Grid cell size (float)\n"
          "    grid_dims: (2,) int64 tensor of (nx, ny) cells\n"
          "    cell_polygon_indices: CSR values for polygon geometry IDs per cell\n"
          "    cell_polygon_offsets: CSR offsets for polygons per cell\n"
          "    polygon_segment_ranges: (R, 2) int64 tensor of [start, end) segment indices per ring\n"
          "    polygon_geom_indices: (R,) int64 tensor of geometry index per ring\n"
          "    geom_ring_offsets: (G_poly+1,) int64 CSR offsets for rings per polygon geometry\n"
          "    debug: Enable debug output (default: False)\n"
          "\n"
          "Returns:\n"
          "    Tuple of (particle_indices, geometry_indices, distances)\n",
          pybind11::arg("query_points"),
          pybind11::arg("segment_starts"),
          pybind11::arg("segment_ends"),
          pybind11::arg("segment_to_geom"),
          pybind11::arg("point_coords"),
          pybind11::arg("point_to_geom"),
          pybind11::arg("num_geometries"),
          pybind11::arg("cell_segment_indices"),
          pybind11::arg("cell_offsets"),
          pybind11::arg("cell_geom_indices"),
          pybind11::arg("cell_geom_offsets"),
          pybind11::arg("cell_point_indices"),
          pybind11::arg("cell_point_offsets"),
          pybind11::arg("cell_point_geom_indices"),
          pybind11::arg("cell_point_geom_offsets"),
          pybind11::arg("grid_origin"),
          pybind11::arg("cell_size"),
          pybind11::arg("grid_dims"),
          pybind11::arg("cell_polygon_indices"),
          pybind11::arg("cell_polygon_offsets"),
          pybind11::arg("polygon_segment_ranges"),
          pybind11::arg("polygon_geom_indices"),
          pybind11::arg("geom_ring_offsets"),
          pybind11::arg("debug") = false);
    m.def("point_in_polygon_sparse_cuda", &point_in_polygon_sparse_cuda,
          "Sparse point-in-polygon test on candidate (point, polygon) pairs");
}

}  // namespace robot::geometry
