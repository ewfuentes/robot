#include "experimental/beacon_sim/ethans_super_cool_planner.hh"

#include <iterator>

namespace robot::experimental::beacon_sim {

RolloutFunctionType make_rollout_function(
    const std::function<Candidate(const Candidate&)>& step_candidate,
    const std::function<bool(const Candidate&, const int)>& terminate_rollout,
    const RollOutArgs& roll_out_args) {
    return [&step_candidate, &terminate_rollout, &roll_out_args](const Candidate& candidate){
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

    };
}

CullingFunctionType make_cull_the_heard_function(
    const std::function<double(const Candidate&)>& scoring_function,
    const CullingArgs& cullingArgs) {

    return [&scoring_function, &cullingArgs](const std::vector<Candidate>& candidates) {
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
        if (num_best_survivors > 0) {  // evaluate each candidate's score
            std::vector<std::pair<double, Candidate>> scored_candidates;
            for (unsigned int i = 0; i < candidates.size(); i++) {
                if (used[i]) {
                    continue;
                }
                scored_candidates.push_back(
                    std::make_pair(scoring_function(candidates[i]), candidates[i]));
            }
            // sort the candidates by score
            std::sort(scored_candidates.begin(), scored_candidates.end(),
                    [](const auto& a, const auto& b) {
                        return a.first > b.first;  // high score is better
                    });
            // take the best candidates
            for (unsigned int i = 0; i < num_best_survivors; i++) {
                survivors.push_back(scored_candidates[i].second);
            }
        }

        return survivors;
    };
}

std::vector<int> plan(
    const Candidate& start_candidate,
    const RolloutFunctionType& rollout_function,
    const CullingFunctionType& culling_function,
    const GoalScoreFunctionType& goal_score_function,
    const PlanningArgs& planning_args) {

    auto candidates = std::vector<Candidate>(planning_args.num_candidates, start_candidate);
    Candidate the_winning_candidate;
    float the_winning_score = std::numeric_limits<float>::min();

    bool terminate = false;
    while (!terminate) {
        std::vector<Candidate> successors;
        // rollout each of the candidates
        for (const auto& candidate : candidates) {
            const auto& new_rollouts = rollout_function( candidate ); // produce more candidates
            std::copy(new_rollouts.begin(), new_rollouts.end(), std::back_inserter(successors));
        }
        // check the goal score of each candidate
        for (const auto& candidate : successors) {
            if (goal_score_function(candidate) > the_winning_score) {
                the_winning_candidate = candidate;
                the_winning_score = goal_score_function(candidate);
            }
        }
        // reduce the number of candidates
        candidates = culling_function(candidates);
    } 

    // find the best candidate
    return the_winning_candidate.path_history;
}

}  // namespace robot::experimental::beacon_sim
