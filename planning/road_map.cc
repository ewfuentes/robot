

#include "planning/road_map.hh"

namespace robot::planning {

RoadMap::RoadMap(std::vector<Eigen::Vector2d> points, Eigen::MatrixXd adj,
                 std::optional<StartGoalPair> start_goal_pair)
    : points_(std::move(points)), adj_(std::move(adj)) {
    if (!start_goal_pair.has_value()) {
        start_goal_ = std::nullopt;
    }

    add_start_goal(*start_goal_pair);
}

void RoadMap::add_start_goal(const StartGoalPair &start_goal_pair) {
    std::vector<int> start_neighbors;
    std::vector<int> goal_neighbors;

    for (int i = 0; i < static_cast<int>(points_.size()); i++) {
        const Eigen::Vector2d &pt = points_.at(i);
        if ((pt - start_goal_pair.start).norm() < start_goal_pair.connection_radius_m) {
            start_neighbors.push_back(i);
        }

        if ((pt - start_goal_pair.goal).norm() < start_goal_pair.connection_radius_m) {
            goal_neighbors.push_back(i);
        }
    }

    start_goal_ = {
        .start = start_goal_pair.start,
        .goal = start_goal_pair.goal,
        .start_neighbors = std::move(start_neighbors),
        .goal_neighbors = std::move(goal_neighbors),
    };
}
}  // namespace robot::planning
