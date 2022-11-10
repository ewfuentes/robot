
#pragma once

#include <functional>
#include <iostream>
#include <optional>

#include "planning/probabilistic_road_map.hh"

namespace robot::planning {

struct Belief {
    Eigen::Vector2d mean;
    Eigen::Matrix2d cov;
};

struct BRMPlan {
    static constexpr int INITIAL_BELIEF_NODE_IDX = -1;
    static constexpr int GOAL_BELIEF_NODE_IDX = -2;
    std::vector<int> nodes;
    std::vector<Belief> beliefs;
};

std::ostream &operator<<(std::ostream &out, const Belief &belief);
std::ostream &operator<<(std::ostream &out, const BRMPlan &plan);

using BeliefUpdater =
    std::function<Belief(const Belief &initial_belief, const int start_idx, const int end_idx)>;
std::optional<BRMPlan> plan(const RoadMap &road_map, const Belief &initial_belief,
                            const BeliefUpdater &belief_updater, const Eigen::Vector2d &goal_state);
}  // namespace robot::planning
