#include "experimental/beacon_sim/ethans_super_cool_planner.hh"
#include <iterator>

namespace robot::experimental::beacon_sim {

//        std::vector<planning::Successor<int>> out;
//        if (node_idx == START_IDX || node_idx == GOAL_IDX) {
//            for (int i = 0; i < static_cast<int>(road_map.points.size()); i++) {
//                const Eigen::Vector2d &pt_in_local = road_map.points.at(i);
//                const double dist_m = (pt_in_local - start_in_local).norm();
//                if (dist_m < start_goal_connection_radius_m) {
//                    out.push_back({.state = i, .edge_cost = dist_m});
//                }
//            }
//        } else {
//            const Eigen::Vector2d &curr_pt_in_local = road_map.points.at(node_idx);
//            for (int i = 0; i < static_cast<int>(road_map.points.size()); i++) {
//                if (road_map.adj(i, node_idx)) {
//                    const Eigen::Vector2d &other_in_local = road_map.points.at(i);
//                    const double dist_m = (curr_pt_in_local - other_in_local).norm();
//                    out.push_back({.state = i, .edge_cost = dist_m});
//                }
//            }
//
//            const double dist_to_goal_m = (curr_pt_in_local - goal_in_local).norm();
//            if (dist_to_goal_m < start_goal_connection_radius_m) {
//                out.push_back({.state = GOAL_IDX, .edge_cost = dist_to_goal_m});
//            }
//        }
//        return out;

std::vector<Candidate> rollout(
    [[maybe_unused]] const planning::RoadMap& map,
    std::function<bool(const Candidate&, int)> terminate_rollout,
    [[maybe_unused]] const Candidate& candidate,
    [[maybe_unused]] std::function<std::vector<planning::Successor<int>>(planning::Successor<int>)>
        successor_function,
    [[maybe_unused]] const planning::BeliefUpdater<RobotBelief>& belief_updater,
    [[maybe_unused]] const RollOutArgs& roll_out_args) {
    // what node are we on

    std::vector<Candidate> successors;
    [[maybe_unused]] int position = candidate.path_history.back();

    for (int rollout_i = 0; rollout_i < roll_out_args.num_roll_outs; rollout_i++) {
        int num_steps = 0;
        Candidate start = candidate;
        while (not terminate_rollout(start, num_steps)) {
            // pick an action

            // traverse the edge

            num_steps++;
        }

        successors.push_back(start);
    }

    return successors;
}

//std::vector<Candidate> cull_the_heard(
//    const std::vector<Candidate>& candidates,
//    [[maybe_unused]] const std::function<double(const Candidate&)>& scoring_function,
//    const CullingArgs& cullingArgs) {
//    // pick randomly from the candidates, without replacement
//    std::vector<Candidate> survivors;
//    std::vector<bool> used(candidates.size(), false);
//    for (int i = 0; i < cullingArgs.num_candidates; i++) {
//        int index = rand() % candidates.size();
//        while (used[index]) {
//            index = rand() % candidates.size();
//        }
//        used[index] = true;
//        survivors.push_back(candidates[index]);
//    }
//
//    return survivors;
//}
//
//std::function<double(const Candidate&)> make_scoring_function(
//    const planning::RoadMap& map, const Eigen::Vector2d& goal_state,
//    const planning::BeliefUpdater<RobotBelief>& belief_updater, const std::vector<int>& beacons) {
//    return [map, goal_state, belief_updater, beacons]([[maybe_unused]] const Candidate& candidate) {
//        // TODO
//        return 0.0;
//    };
//}
//
//std::vector<int> plan(
//    const planning::RoadMap& map, const Candidate& start_state, const Eigen::Vector2d& goal_state,
//    const planning::BeliefUpdater<RobotBelief>& belief_updater,
//    const std::function<double(const Candidate&)>& scoring_function,
//    const std::function<std::vector<Candidate>(const planning::RoadMap&, const Candidate&,
//                                               const planning::BeliefUpdater<RobotBelief>&,
//                                               const RollOutArgs&)>& rollout_function,
//    const std::function<std::vector<Candidate>(const std::vector<Candidate>&,
//                                               const std::function<double(const Candidate&)>&,
//                                               const CullingArgs&)>& culling_function,
//    const PlanningArgs& planning_args) {
//    auto candidates = std::vector<Candidate>(planning_args.num_candidates, start_state);
//
//    std::vector<Candidate> successful_candidates;  // list of candidates that have reached the goal
//
//    bool terminate = false;
//    while (!terminate) {
//
//        std::vector<Candidate> successors;
//    for (const auto& candidate : candidates) {
//            const auto &new_rollouts = rollout(map, candidate, rollout_function, planning_args.rollout_args);
//    std::copy(new_rollouts.begin(), new_rollouts.end(), std::back_inserter(successors));
//    }
//    for (const auto& candidate : successors) {
//        // TODO
//    }
//
//    candidates = cull_the_heard(candidates, scoring_function, planning_args.culling_args);
//    }
//
//    // find the best candidate
//    const auto &best_candidate = candidates.front();
//    return best_candidate.path_history;
//}

}  // namespace robot::experimental::beacon_sim
