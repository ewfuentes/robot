
#include "common/time/robot_time.hh"
#include "experimental/beacon_sim/belief_road_map_planner.hh"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace robot::experimental::beacon_sim {

template <typename T>
constexpr void bind_brm_plan(const auto &m, const std::string &type) {
    const std::string type_name = "BeliefRoadMapPlan" + type;
    py::class_<planning::BRMPlan<T>>(m, type_name.c_str())
        .def_readwrite("nodes", &planning::BRMPlan<T>::nodes)
        .def_readwrite("beliefs", &planning::BRMPlan<T>::beliefs);
}

PYBIND11_MODULE(belief_road_map_planner_python, m) {
    py::module_::import("experimental.beacon_sim.robot_belief_python");
    bind_brm_plan<RobotBelief>(m, "RobotBelief");
    py::class_<BeliefRoadMapOptions>(m, "BeliefRoadMapOptions")
        .def(py::init<>())
        .def(py::init<double, std::optional<double>, int,
                      std::optional<time::RobotTimestamp::duration>>(),
             "max_sensor_range_m"_a, "uncertainty_tolerance"_a, "max_num_edge_transforms"_a,
             "timeout"_a)
        .def_readwrite("max_sensor_range_m", &BeliefRoadMapOptions::max_sensor_range_m)
        .def_readwrite("uncertainty_tolerance", &BeliefRoadMapOptions::uncertainty_tolerance)
        .def_readwrite("max_num_edge_transforms", &BeliefRoadMapOptions::max_num_edge_transforms)
        .def_readwrite("timeout", &BeliefRoadMapOptions::timeout);

    m.def("compute_belief_road_map_plan", &compute_belief_road_map_plan);
}
}  // namespace robot::experimental::beacon_sim
