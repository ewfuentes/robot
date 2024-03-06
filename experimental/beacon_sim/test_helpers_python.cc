
#include "experimental/beacon_sim/test_helpers.hh"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace robot::experimental::beacon_sim {
PYBIND11_MODULE(test_helpers_python, m) {
    py::module_::import("experimental.beacon_sim.correlated_beacons_python");
    py::module_::import("planning.probabilistic_road_map_python");
    m.def("create_grid_environment", &create_grid_environment);
    m.def("create_diamond_environment", &create_diamond_environment);
    m.def("create_stress_test_environment", &create_stress_test_environment);
}
}  // namespace robot::experimental::beacon_sim
