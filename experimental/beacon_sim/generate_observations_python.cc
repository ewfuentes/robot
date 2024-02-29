
#include "experimental/beacon_sim/generate_observations.hh"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include <sstream>

namespace py = pybind11;

namespace robot::experimental::beacon_sim {
PYBIND11_MODULE(generate_observations_python, m) {
    py::class_<BeaconObservation>(m, "BeaconObservation")
        .def(py::init<std::optional<int>, std::optional<double>, std::optional<double>>())
        .def_readwrite("maybe_id", &BeaconObservation::maybe_id)
        .def_readwrite("maybe_range_m", &BeaconObservation::maybe_range_m)
        .def_readwrite("maybe_bearing_rad", &BeaconObservation::maybe_bearing_rad)
        .def("__repr__", [](const BeaconObservation &obs){
             std::ostringstream ss;
            ss << "<BeaconObservation id:"
                << (obs.maybe_id.has_value() ? std::to_string(obs.maybe_id.value()) : "Unknown")
                << " range_m: " << (obs.maybe_range_m.has_value() ? std::to_string(obs.maybe_range_m.value()) : "Unknown")
                << " bearing_rad: " << (obs.maybe_bearing_rad.has_value() ? std::to_string(obs.maybe_bearing_rad.value()) : "Unknown")
                << ">";
            return ss.str();
        });
}
}  // namespace robot::experimental::beacon_sim
