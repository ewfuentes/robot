

#include "planning/road_map.hh"

#include "common/check.hh"

namespace robot::planning {

RoadMap::RoadMap(std::vector<Eigen::Vector2d> points, Eigen::MatrixXd adj,
                 std::optional<StartGoalPair> start_goal_pair)
    : points_(std::move(points)), adj_(std::move(adj)) {
    if (!start_goal_pair.has_value()) {
        start_goal_ = std::nullopt;
    }

    if (start_goal_pair.has_value()) {
        add_start_goal(*start_goal_pair);
    }
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

    ROBOT_CHECK(!start_neighbors.empty(), "No roadmap points near start",
                start_goal_pair.start.transpose(), start_goal_pair.connection_radius_m);
    ROBOT_CHECK(!goal_neighbors.empty(), "No roadmap points near goal",
                start_goal_pair.goal.transpose(), start_goal_pair.connection_radius_m);

    start_goal_ = {
        .start = start_goal_pair.start,
        .goal = start_goal_pair.goal,
        .start_neighbors = std::move(start_neighbors),
        .goal_neighbors = std::move(goal_neighbors),
    };
}

const Eigen::Vector2d &RoadMap::point(const int idx) const {
    ROBOT_CHECK(idx >= RoadMap::GOAL_IDX, "Invalid index");
    if (idx >= 0) {
        return points_.at(idx);
    } else if (idx == RoadMap::START_IDX) {
        return start_goal_.value().start;
    } else {
        return start_goal_.value().goal;
    }
}

std::vector<std::tuple<int, Eigen::Vector2d>> RoadMap::neighbors(const int idx) const {
    ROBOT_CHECK(idx >= RoadMap::GOAL_IDX, "Invalid index");
    std::vector<std::tuple<int, Eigen::Vector2d>> out;
    if (idx >= 0) {
        for (int other_idx = 0; other_idx < static_cast<int>(points_.size()); other_idx++) {
            if (adj_(idx, other_idx)) {
                out.push_back(std::make_tuple(other_idx, points_.at(other_idx)));
            }
        }

        if (start_goal_.has_value()) {
            {
                const auto iter = std::find(start_goal_->start_neighbors.begin(),
                                            start_goal_->start_neighbors.end(), idx);
                if (iter != start_goal_->start_neighbors.end()) {
                    out.push_back(std::make_tuple(RoadMap::START_IDX, start_goal_->start));
                }
            }

            {
                const auto iter = std::find(start_goal_->goal_neighbors.begin(),
                                            start_goal_->goal_neighbors.end(), idx);
                if (iter != start_goal_->goal_neighbors.end()) {
                    out.push_back(std::make_tuple(RoadMap::GOAL_IDX, start_goal_->goal));
                }
            }
        }
    } else {
        const std::vector<int> &neighbors = idx == RoadMap::START_IDX
                                                ? start_goal_.value().start_neighbors
                                                : start_goal_.value().goal_neighbors;

        for (const int neighbor_idx : neighbors) {
            out.push_back(std::make_tuple(neighbor_idx, points_.at(neighbor_idx)));
        }
    }
    return out;
}

}  // namespace robot::planning
