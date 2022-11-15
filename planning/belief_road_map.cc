
#include "planning/belief_road_map.hh"

namespace robot::planning::detail {
int find_nearest_node_idx(const RoadMap &road_map, const Eigen::Vector2d &state) {
    const auto iter =
        std::min_element(road_map.points.begin(), road_map.points.end(),
                         [&state](const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
                             return (a - state).squaredNorm() < (b - state).squaredNorm();
                         });
    return std::distance(road_map.points.begin(), iter);
}
}  // namespace robot::planning
