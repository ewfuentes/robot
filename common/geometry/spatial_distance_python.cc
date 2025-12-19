#include "common/geometry/spatial_distance.hh"
#include "torch/extension.h"

namespace robot::geometry {

PYBIND11_MODULE(spatial_distance_python, m) {
    m.doc() = "GPU-accelerated spatial distance queries with CUDA";
    m.def("query_distances_cuda", &query_distances_cuda,
          "Query distances from points to geometries using CUDA kernel with spatial index");
}

}  // namespace robot::geometry
