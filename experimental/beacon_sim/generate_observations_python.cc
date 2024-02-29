
#include "experimental/beacon_sim/generate_observations.hh"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace robot::experimental::beacon_sim {
PYBIND11_MODULE(generate_observations_python, m) {
    py::class_<BeaconObservation>(m, "BeaconObservation")
        .def(py::init<std::optional<int>, std::optional<double>, std::optional<double>>())
        .def_readwrite("maybe_id", &BeaconObservation::maybe_id)
        .def_readwrite("maybe_range_m", &BeaconObservation::maybe_range_m)
        .def_readwrite("maybe_bearing_rad", &BeaconObservation::maybe_bearing_rad);
}
}  // namespace robot::experimental::beacon_sim
