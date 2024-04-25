
#include <random>

#include "experimental/beacon_sim/beacon_potential.hh"
#include "experimental/beacon_sim/beacon_potential_to_proto.hh"
#include "experimental/beacon_sim/correlated_beacon_potential.hh"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using namespace py::literals;

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
        .def_static(
            "correlated_beacon_potential",
            [](const double p_present, const double p_beacon_given_present,
               std::vector<int> &members) -> BeaconPotential {
                return CorrelatedBeaconPotential{.p_present = p_present,
                                                 .p_beacon_given_present = p_beacon_given_present,
                                                 .members = members};
            },
            "p_present"_a, "p_beacon_given_present"_a, "members"_a)
        .def_static(
            "combined_potential",
            [](std::vector<BeaconPotential> &pots) -> BeaconPotential {
                return CombinedPotential(pots);
            },
            "pots"_a)
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
        .def("conditioned_on", &BeaconPotential::conditioned_on)
        .def("sample",
             [](const BeaconPotential &pot, const int seed) {
                 std::mt19937 gen(seed);
                 const std::vector<int> present_beacons = pot.sample(make_in_out(gen));
                 std::unordered_map<int, bool> out;
                 for (const int beacon_id : pot.members()) {
                     out[beacon_id] = false;
                 }
                 for (const int beacon_id : present_beacons) {
                     out[beacon_id] = true;
                 }
                 return out;
             })
        .def("to_proto_string", [](const BeaconPotential &self) {
            proto::BeaconPotential proto;
            pack_into(self, &proto);
            std::string out;
            proto.SerializeToString(&out);
            return py::bytes(out);
        });
}
}  // namespace robot::experimental::beacon_sim
