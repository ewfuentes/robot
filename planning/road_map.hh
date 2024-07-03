#pragma once

#include <optional>
#include <vector>

#include "Eigen/Core"

namespace robot::planning {

struct StartGoalPair {
    Eigen::Vector2d start;
    Eigen::Vector2d goal;
    double connection_radius_m;
};

class RoadMap {
   public:
    static constexpr int START_IDX = -1;
    static constexpr int GOAL_IDX = -2;

    RoadMap(std::vector<Eigen::Vector2d> points ,Eigen::MatrixXd adj,
            std::optional<StartGoalPair> start_goal_pair = std::nullopt);

    void add_start_goal(const StartGoalPair &start_goal_pair);

    const std::vector<Eigen::Vector2d> &points() const { return points_; }
    const Eigen::MatrixXd &adj() const { return adj_; }
    bool has_start_goal() const { return start_goal_.has_value(); }

    // Throws if the index is invalid
    const Eigen::Vector2d &point(const int idx) const;

    // Throws if the index is invalid
    std::vector<std::tuple<int, Eigen::Vector2d>> neighbors(const int idx) const;

   private:
    struct StartGoalBlock {
        Eigen::Vector2d start;
        Eigen::Vector2d goal;
        std::vector<int> start_neighbors;
        std::vector<int> goal_neighbors;
    };
    std::optional<StartGoalBlock> start_goal_;

    // The location of each roadmap point
    std::vector<Eigen::Vector2d> points_;

    // An adjacency matrix where the i'th column contains the connectivity to other nodes
    Eigen::MatrixXd adj_;
};

}  // namespace robot::planning
