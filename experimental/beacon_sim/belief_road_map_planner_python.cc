
#include "common/time/robot_time.hh"
#include "experimental/beacon_sim/belief_road_map_planner.hh"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/chrono.h"

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
    bind_brm_plan<LandmarkRobotBelief>(m, "LandmarkRobotBelief");

    py::class_<ExpectedBeliefPlanResult>(m, "ExpectedBeliefPlanResult")
        .def(py::init<>())
        .def(py::init<std::vector<int>, double>(), "nodes"_a, "log_probability_mass_tracked"_a)
        .def_readwrite("nodes", &ExpectedBeliefPlanResult::nodes)
        .def_readwrite("log_probability_mass_tracked",
                       &ExpectedBeliefPlanResult::log_probability_mass_tracked);

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

    auto landmark_brm_options =
        py::class_<LandmarkBeliefRoadMapOptions>(m, "LandmarkBeliefRoadMapOptions")
            .def(py::init<>())
            .def(py::init<double, LandmarkBeliefRoadMapOptions::UncertaintySizeOptions,
                          std::optional<LandmarkBeliefRoadMapOptions::SampledBeliefOptions>,
                          std::optional<time::RobotTimestamp::duration>>(),
                 "max_sensor_range_m"_a, "uncertainty_size_options"_a, "sampled_belief_options"_a,
                 "timeout"_a)
            .def_readwrite("max_sensor_range_m", &LandmarkBeliefRoadMapOptions::max_sensor_range_m)
            .def_readwrite("uncertainty_size_options",
                           &LandmarkBeliefRoadMapOptions::uncertainty_size_options)
            .def_readwrite("sampled_belief_options",
                           &LandmarkBeliefRoadMapOptions::sampled_belief_options)
            .def_readwrite("timeout", &LandmarkBeliefRoadMapOptions::timeout);

    py::class_<LandmarkBeliefRoadMapOptions::SampledBeliefOptions>(landmark_brm_options,
                                                                   "SampledBeliefOptions")
        .def(py::init<>())
        .def(py::init<int, int>(), "max_num_components"_a, "seed"_a)
        .def_readwrite("max_num_components",
                       &LandmarkBeliefRoadMapOptions::SampledBeliefOptions::max_num_components)
        .def_readwrite("seed", &LandmarkBeliefRoadMapOptions::SampledBeliefOptions::seed);

    py::class_<LandmarkBeliefRoadMapOptions::ExpectedDeterminant>(landmark_brm_options,
                                                                  "ExpectedDeterminant")
        .def(py::init<>());

    py::class_<LandmarkBeliefRoadMapOptions::ValueAtRiskDeterminant>(landmark_brm_options,
                                                                     "ValueAtRiskDeterminant")
        .def(py::init<>())
        .def(py::init<double>(), "percentile"_a)
        .def_readwrite("percentile",
                       &LandmarkBeliefRoadMapOptions::ValueAtRiskDeterminant::percentile);

    py::class_<LandmarkBeliefRoadMapOptions::ProbMassInRegion>(landmark_brm_options,
                                                               "ProbMassInRegion")
        .def(py::init<>())
        .def(py::init<double, double, double>(), "position_x_half_width_m"_a,
             "position_y_half_width_m"_a, "heading_half_width_rad"_a)
        .def_readwrite("position_x_half_width_m",
                       &LandmarkBeliefRoadMapOptions::ProbMassInRegion::position_x_half_width_m)
        .def_readwrite("position_y_half_width_m",
                       &LandmarkBeliefRoadMapOptions::ProbMassInRegion::position_y_half_width_m)
        .def_readwrite("heading_half_width_rad",
                       &LandmarkBeliefRoadMapOptions::ProbMassInRegion::heading_half_width_rad);

    py::class_<ExpectedBeliefRoadMapOptions>(m, "ExpectedBeliefRoadMapOptions")
        .def(py::init<>())
        .def(py::init<int, int, BeliefRoadMapOptions>(), "num_configuration_samples"_a, "seed"_a,
             "brm_options"_a)
        .def_readwrite("num_configuration_samples",
                       &ExpectedBeliefRoadMapOptions::num_configuration_samples)
        .def_readwrite("seed", &ExpectedBeliefRoadMapOptions::seed)
        .def_readwrite("brm_options", &ExpectedBeliefRoadMapOptions::brm_options);

    m.def("compute_belief_road_map_plan", &compute_belief_road_map_plan);
    m.def("compute_landmark_belief_road_map_plan", &compute_landmark_belief_road_map_plan);
    m.def("compute_expected_belief_road_map_plan", &compute_expected_belief_road_map_plan);
}
}  // namespace robot::experimental::beacon_sim
