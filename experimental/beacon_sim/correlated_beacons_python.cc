

#include <pybind11/detail/common.h>

#include "experimental/beacon_sim/correlated_beacons.hh"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace robot::experimental::beacon_sim {
PYBIND11_MODULE(correlated_beacons_python, m) {
    m.doc() = "Correlated Beacons";

    py::class_<BeaconClique>(m, "BeaconClique")
        .def(py::init<double, double, std::vector<int>>(), "p_beacon"_a, "p_no_beacons"_a,
             "members"_a)
        .def_readwrite("p_beacon", &BeaconClique::p_beacon)
        .def_readwrite("p_no_beacons", &BeaconClique::p_no_beacons)
        .def_readwrite("members", &BeaconClique::members);

    py::class_<BeaconPotential>(m, "BeaconPotential")
        .def(py::init<Eigen::MatrixXd, double, std::vector<int>>())
        .def("log_prob", &BeaconPotential::log_prob)
        .def("__mul__", &BeaconPotential::operator*);

    m.def("create_correlated_beacons", create_correlated_beacons);
}
}  // namespace robot::experimental::beacon_sim
