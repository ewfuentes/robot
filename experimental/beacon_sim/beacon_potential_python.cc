
#include "experimental/beacon_sim/beacon_potential.hh"
#include "experimental/beacon_sim/beacon_potential_to_proto.hh"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace robot::experimental::beacon_sim {
PYBIND11_MODULE(beacon_potential_python, m) {
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
        .def(py::init<>())
        .def_static("from_proto_string",
                    [](const std::string &proto_string) -> BeaconPotential {
                        proto::BeaconPotential proto;
                        proto.ParseFromString(proto_string);
                        return unpack_from(proto);
                    })
        .def("log_prob",
             py::overload_cast<const std::unordered_map<int, bool> &, bool>(
                 &BeaconPotential::log_prob, py::const_),
             py::arg("assignment"), py::arg("allow_partial_assignment") = false)
        .def("log_prob",
             py::overload_cast<const std::vector<int> &>(&BeaconPotential::log_prob, py::const_))
        .def("__mul__",
             py::overload_cast<const BeaconPotential &, const BeaconPotential &>(operator*))
        .def("log_marginals", &BeaconPotential::log_marginals)
        .def("members", &BeaconPotential::members)
        .def("to_proto_string", [](const BeaconPotential &self) {
            proto::BeaconPotential proto;
            pack_into(self, &proto);
            std::string out;
            proto.SerializeToString(&out);
            return py::bytes(out);
        });
}
}  // namespace robot::experimental::beacon_sim
