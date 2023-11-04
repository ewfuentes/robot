#include "experimental/beacon_sim/ethans_super_cool_planner.hh"
#include <iterator>

namespace robot::experimental::beacon_sim {

std::vector<Candidate> rollout(
    const std::function<Candidate(const Candidate&)>& step_candidate,
    const std::function<bool(const Candidate&, const int)>& terminate_rollout,
    const Candidate& candidate,
    const RollOutArgs& roll_out_args) {

    std::vector<Candidate> successors;

    for (unsigned int rollout_i = 0; rollout_i < roll_out_args.num_roll_outs; rollout_i++) {
        int num_steps = 0;
        Candidate current_candidate = candidate;
        while (!terminate_rollout(current_candidate, num_steps)) {
            // step the candidate
            current_candidate = step_candidate(current_candidate);
            num_steps++;
        }
        successors.push_back(current_candidate);
    }

    return successors;
}

std::vector<Candidate> cull_the_heard(
   const std::vector<Candidate>& candidates,
   const std::function<double(const Candidate&)>& scoring_function,
   const CullingArgs& cullingArgs) {

   std::vector<Candidate> survivors;
   std::vector<bool> used(candidates.size(), false);

   unsigned int num_random_survivors = cullingArgs.num_survivors * cullingArgs.entropy_proxy;
   auto num_best_survivors = cullingArgs.num_survivors - num_random_survivors;

   // pick the random survivors
   for (unsigned int i = 0; i < num_random_survivors; i++) {
       int index = rand() % candidates.size();
       while (used[index]) {
           index = rand() % candidates.size();
       }
       used[index] = true;
       survivors.push_back(candidates[index]);
   }

   // pick the best survivors
    if (num_best_survivors > 0) { // evaluate each candidate's score
        std::vector<std::pair<double, Candidate>> scored_candidates;
        for (unsigned int i = 0; i < candidates.size(); i++) {
            if (used[i]) {
                continue;
            }
            scored_candidates.push_back(std::make_pair(scoring_function(candidates[ i ]), candidates[ i ]));
        }
        // sort the candidates by score
        std::sort(scored_candidates.begin(), scored_candidates.end(), [](const auto& a, const auto& b) {
             return a.first > b.first; // high score is better
        });
        // take the best candidates
        for (unsigned int i = 0; i < num_best_survivors; i++) {
             survivors.push_back(scored_candidates[i].second);
        }
   }

   return survivors;

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

