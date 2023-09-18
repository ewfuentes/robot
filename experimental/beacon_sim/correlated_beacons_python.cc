

#include <sstream>

#include "experimental/beacon_sim/correlated_beacons.hh"
#include "pybind11/eigen.h"
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

    py::class_<LogMarginal>(m, "LogMarginal")
        .def_readwrite("present_beacons", &LogMarginal::present_beacons)
        .def_readwrite("log_marginal", &LogMarginal::log_marginal)
        .def("__repr__", [](const LogMarginal &self) {
            std::ostringstream oss;
            oss << "<LogMarginal present_beacons=[";
            for (int i = 0; i < static_cast<int>(self.present_beacons.size()); i++) {
                oss << self.present_beacons.at(i);
                if (i < static_cast<int>(self.present_beacons.size()) - 1) {
                    oss << ", ";
                }
            }
            oss << "] log_marginal=" << self.log_marginal << ">";
            return oss.str();
        });

    py::class_<BeaconPotential>(m, "BeaconPotential")
        .def(py::init<Eigen::MatrixXd, double, std::vector<int>>())
        .def("log_prob", py::overload_cast<const std::unordered_map<int, bool> &>(
                             &BeaconPotential::log_prob, py::const_))
        .def("log_prob",
             py::overload_cast<const std::vector<int> &>(&BeaconPotential::log_prob, py::const_))
        .def("__mul__", &BeaconPotential::operator*)
        .def("compute_log_marginals", &BeaconPotential::compute_log_marginals);

    m.def("create_correlated_beacons", create_correlated_beacons);
}
}  // namespace robot::experimental::beacon_sim
