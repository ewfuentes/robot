
#include "experimental/beacon_sim/robot_belief.hh"
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace robot::experimental::beacon_sim {
PYBIND11_MODULE(robot_belief_python, m) {
    py::class_<RobotBelief>(m, "RobotBelief")
        .def(py::init<>())
        .def_readwrite("local_from_robot", &RobotBelief::local_from_robot)
        .def_readwrite("cov_in_robot", &RobotBelief::cov_in_robot);

    auto landmark_robot_belief =
        py::class_<LandmarkRobotBelief>(m, "LandmarkRobotBelief")
            .def_readwrite("local_from_robot", &LandmarkRobotBelief::local_from_robot)
            .def_readwrite("log_probability_mass_tracked",
                           &LandmarkRobotBelief::log_probability_mass_tracked)
            .def_readwrite("belief_from_config", &LandmarkRobotBelief::belief_from_config);

    py::class_<LandmarkRobotBelief::LandmarkConditionedRobotBelief>(
        landmark_robot_belief, "LandmarkConditionedRobotBelief")
        .def_readwrite("cov_in_robot",
                       &LandmarkRobotBelief::LandmarkConditionedRobotBelief::cov_in_robot)
        .def_readwrite("log_config_prob",
                       &LandmarkRobotBelief::LandmarkConditionedRobotBelief::log_config_prob);
}
}  // namespace robot::experimental::beacon_sim
