#include "experimental/beacon_sim/ethans_super_cool_planner.hh"
#include <iterator>

namespace robot::experimental::beacon_sim {

std::vector<Candidate> rollout(
    [[maybe_unused]] const planning::RoadMap& map,
    const std::function<bool(const Candidate&, const int)>& terminate_rollout,
    const Candidate& candidate,
    const std::function<std::vector<planning::Successor<int>>(const int)>&
        successor_function,
    const planning::BeliefUpdater<RobotBelief>& belief_updater,
    const RollOutArgs& roll_out_args) {
    // what node are we on

    std::vector<Candidate> successors;


    for (unsigned int rollout_i = 0; rollout_i < roll_out_args.num_roll_outs; rollout_i++) {
        int num_steps = 0;
        Candidate start = candidate;
        int current_index = candidate.path_history.back();

        while (!terminate_rollout(start, num_steps)) {
            // generate actions
            auto successors = successor_function(current_index);
            if (successors.empty()) {
                std::cout << " no successors " << std::endl;
                break;
            }
            // pick an action
            int action_index = rand() % successors.size();
            // traverse the edge
            start.belief = belief_updater(start.belief, current_index, successors[action_index].state);
            start.path_history.push_back(successors[action_index].state);
            current_index = successors[action_index].state;
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

