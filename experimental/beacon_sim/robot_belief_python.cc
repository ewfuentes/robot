
#include "experimental/beacon_sim/robot_belief.hh"
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace robot::experimental::beacon_sim {
PYBIND11_MODULE(robot_belief_python, m) {
    py::class_<RobotBelief>(m, "RobotBelief")
        .def_readwrite("local_from_robot", &RobotBelief::local_from_robot)
        .def_readwrite("cov_in_robot", &RobotBelief::cov_in_robot);
}
}  // namespace robot::experimental::beacon_sim
