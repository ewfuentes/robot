#include "common/geometry/spatial_distance.hh"
#include "torch/extension.h"

namespace robot::geometry {

PYBIND11_MODULE(spatial_distance_python, m) {
    m.doc() = "GPU-accelerated spatial distance queries with CUDA";
    m.def("query_distances_cuda", &query_distances_cuda,
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
          "\n"
          "Returns:\n"
          "    Tuple of (particle_indices, geometry_indices, distances)\n");
    m.def("point_in_polygon_sparse_cuda", &point_in_polygon_sparse_cuda,
          "Sparse point-in-polygon test on candidate (point, polygon) pairs");
}

}  // namespace robot::geometry
