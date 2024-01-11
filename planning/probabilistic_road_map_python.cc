
#include "planning/probabilistic_road_map.hh"
#include "planning/road_map.hh"
#include "planning/road_map_to_proto.hh"
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace robot::planning {

PYBIND11_MODULE(probabilistic_road_map_python, m) {
    m.doc() = "probabilistic road map";

    py::class_<RoadmapCreationConfig>(m, "RoadmapCreationConfig")
        .def(py::init<>())
        .def_readwrite("seed", &RoadmapCreationConfig::seed)
        .def_readwrite("num_valid_points", &RoadmapCreationConfig::num_valid_points)
        .def_readwrite("desired_node_degree", &RoadmapCreationConfig::desired_node_degree);

    py::class_<MapBounds>(m, "MapBounds")
        .def(py::init<>())
        .def_readwrite("bottom_left", &MapBounds::bottom_left)
        .def_readwrite("top_right", &MapBounds::bottom_left);

    py::class_<StartGoalPair>(m, "StartGoalPair")
        .def(py::init<>())
        .def_readwrite("start", &StartGoalPair::start)
        .def_readwrite("goal", &StartGoalPair::goal)
        .def_readwrite("connection_radius_m", &StartGoalPair::connection_radius_m);

    py::class_<RoadMap>(m, "RoadMap")
        .def_readonly_static("START_IDX", &RoadMap::START_IDX)
        .def_readonly_static("GOAL_IDX", &RoadMap::GOAL_IDX)
        .def(
            py::init<std::vector<Eigen::Vector2d>, Eigen::MatrixXd, std::optional<StartGoalPair>>(),
            "points"_a, "adj"_a, "start_goal_pair"_a)
        .def("add_start_goal", &RoadMap::add_start_goal)
        .def("points", &RoadMap::points)
        .def("adj", &RoadMap::adj)
        .def("has_start_goal", &RoadMap::has_start_goal)
        .def("point", &RoadMap::point)
        .def("neighbors", &RoadMap::neighbors)
        .def("to_proto_string", [](const RoadMap &self) {
            proto::RoadMap proto;
            pack_into(self, &proto);
            std::string out;
            proto.SerializeToString(&out);
            return py::bytes(out);
        });

    m.def("create_road_map", [](const MapBounds &bounds, const RoadmapCreationConfig &config) {
        struct Map {
            MapBounds bounds;

            bool in_free_space(const Eigen::Vector2d &) const { return true; }
            bool in_free_space(const Eigen::Vector2d &, const Eigen::Vector2d &) const {
                return true;
            }
            MapBounds map_bounds() const { return bounds; }
        };
        Map map = {.bounds = bounds};

        return create_road_map(map, config);
    });
};

}  // namespace robot::planning
