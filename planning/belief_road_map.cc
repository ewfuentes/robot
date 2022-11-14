
#include "planning/belief_road_map.hh"

#include "Eigen/Dense"
#include "planning/a_star.hh"

namespace robot::planning {
namespace {
struct BRMSearchState {
    Belief belief;
    int node_idx;
    bool operator==(const BRMSearchState &other) const {
        return node_idx == other.node_idx && (belief.cov - other.belief.cov).norm() < 1e-6;
    }
};

int find_nearest_node_idx(const RoadMap &road_map, const Eigen::Vector2d &state) {
    const auto iter =
        std::min_element(road_map.points.begin(), road_map.points.end(),
                         [&state](const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
                             return (a - state).squaredNorm() < (b - state).squaredNorm();
                         });
    return std::distance(road_map.points.begin(), iter);
}

BRMPlan brm_plan_from_a_star_result(const AStarResult<BRMSearchState> &a_star_result) {
    BRMPlan out;
    for (const auto &plan_node : a_star_result.states) {
        out.nodes.push_back(plan_node.node_idx);
        out.beliefs.push_back(plan_node.belief);
    }
    return out;
}
}  // namespace

std::ostream &operator<<(std::ostream &out, const Belief &belief) {
    out << "Mean: " << belief.mean.transpose() << " cov:" << std::endl;
    out << belief.cov;
    return out;
}

std::ostream &operator<<(std::ostream &out, const BRMPlan &plan) {
    for (int i = 0; i < static_cast<int>(plan.nodes.size()); i++) {
        out << "Node: " << plan.nodes.at(i) << " " << plan.beliefs.at(i) << std::endl;
    }
    return out;
}

std::vector<Successor<BRMSearchState>> successors_for_state(const BRMSearchState &state,
                                                            const RoadMap &road_map,
                                                            const BeliefUpdater &belief_updater,
                                                            const int nearest_to_start_idx,
                                                            const int nearest_to_end_idx) {
    std::vector<Successor<BRMSearchState>> out;
    if (state.node_idx == BRMPlan::INITIAL_BELIEF_NODE_IDX) {
        // If we're at the start node, add a successor to the nearest node
        const Belief new_belief =
            belief_updater(state.belief, state.node_idx, nearest_to_start_idx);
        out.push_back(Successor<BRMSearchState>{
            .state =
                BRMSearchState{
                    .belief = new_belief,
                    .node_idx = nearest_to_start_idx,
                },
            .edge_cost = new_belief.cov.determinant() - state.belief.cov.determinant(),
        });
    } else if (state.node_idx == nearest_to_end_idx) {
        // If we're at the node nearest to the goal pose, add a successor to the goal
        const Belief new_belief =
            belief_updater(state.belief, state.node_idx, BRMPlan::GOAL_BELIEF_NODE_IDX);
        out.push_back(Successor<BRMSearchState>{
            .state =
                BRMSearchState{
                    .belief = new_belief,
                    .node_idx = BRMPlan::GOAL_BELIEF_NODE_IDX,
                },
            .edge_cost = new_belief.cov.determinant() - state.belief.cov.determinant(),
        });
    }
    if (state.node_idx < 0) {
        // We're at the start node or the goal node, we've already added all the successors
        // available
        return out;
    }

    // This step cost is added to the edge cost to give preference to shorter paths
    constexpr double STEP_COST = 1e-6;
    for (int i = 0; i < static_cast<int>(road_map.points.size()); i++) {
        if (road_map.adj(state.node_idx, i)) {
            // Queue up each neighbor
            const Belief new_belief = belief_updater(state.belief, state.node_idx, i);
            out.push_back(Successor<BRMSearchState>{
                .state =
                    BRMSearchState{
                        .belief = new_belief,
                        .node_idx = i,
                    },
                .edge_cost =
                    new_belief.cov.determinant() - state.belief.cov.determinant() + STEP_COST,
            });
        }
    }
    return out;
}

std::optional<BRMPlan> plan(const RoadMap &road_map, const Belief &initial_belief,
                            const BeliefUpdater &belief_updater,
                            const Eigen::Vector2d &goal_state) {
    // Find nearest node to start and end states
    const int nearest_to_start_idx = find_nearest_node_idx(road_map, initial_belief.mean);
    const int nearest_to_end_idx = find_nearest_node_idx(road_map, goal_state);

    const auto successors_func = [nearest_to_start_idx, nearest_to_end_idx, &belief_updater,
                                  &road_map](const BRMSearchState &state) {
        return successors_for_state(state, road_map, belief_updater, nearest_to_start_idx,
                                    nearest_to_end_idx);
    };
    const auto heuristic_func = [](const auto &) { return 0.0; };
    const auto goal_check_func = [](const auto &brm_search_state) {
        return brm_search_state.node_idx == BRMPlan::GOAL_BELIEF_NODE_IDX;
    };

    const auto a_star_result = a_star(
        BRMSearchState{.belief = initial_belief, .node_idx = BRMPlan::INITIAL_BELIEF_NODE_IDX},
        successors_func, heuristic_func, goal_check_func);

    return a_star_result.has_value()
               ? std::make_optional(brm_plan_from_a_star_result(a_star_result.value()))
               : std::nullopt;
}
}  // namespace robot::planning

namespace std {
template <>
struct std::hash<robot::planning::BRMSearchState> {
    size_t operator()(const robot::planning::BRMSearchState &state) const {
        std::hash<double> double_hasher;
        std::hash<int> int_hasher;
        return (double_hasher(state.belief.mean.norm()) << 1) ^
               (double_hasher(state.belief.cov.determinant())) ^ (int_hasher(state.node_idx) << 3);
    }  // namespace std
};
}  // namespace std
