#include "experimental/beacon_sim/ethans_super_cool_planner.hh"
#include "common/check.hh"
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
    const Candidate& start_candidate,
    const CullingArgs& cullingArgs) {

    return [&scoring_function, &cullingArgs, &start_candidate](const std::vector<Candidate>& candidates) {
        std::vector<Candidate> survivors;

        // score candidates, remove all invalid candidates
        std::vector<std::pair<double, Candidate>> scored_candidates;
        for (unsigned int i = 0; i < candidates.size(); i++) {
            double candidate_score = scoring_function(candidates[i]);
            if (candidate_score < 0) {
                continue;
            }
            scored_candidates.push_back(
                std::make_pair(candidate_score, candidates[i]));
        }

        // sort the candidates by score
        std::sort(scored_candidates.begin(), scored_candidates.end(),
                [](const auto& a, const auto& b) {
                    return a.first > b.first;  // high score is better
                });

        std::vector<bool> used(scored_candidates.size(), false);

        CHECK(cullingArgs.entropy_proxy + cullingArgs.reseed_percentage <= 1.0); // can't have more than 100% of the population
        int final_population = std::min(cullingArgs.max_num_survivors, candidates.size());
        unsigned long num_random_survivors = final_population * cullingArgs.entropy_proxy;
        unsigned long num_reseed_survivors = final_population * cullingArgs.reseed_percentage;
        unsigned long num_best_survivors = final_population - num_random_survivors - num_reseed_survivors;
        CHECK(num_best_survivors >= 0);

        // pick the random survivors without replacement 
        int num_unused = scored_candidates.size();
        for (unsigned int i = 0; i < num_random_survivors; i++) {
            int unused_index = rand() % num_unused;
            int index = 0;
            for (unsigned int j = 0; j < scored_candidates.size(); j++) {
                if (!used[j]) {
                    if (index == unused_index) {
                        index = j;
                        break;
                    }
                    index++;
                }
            }

            used[index] = true;
            survivors.push_back(scored_candidates[index].second);
            num_unused--;
            if (num_unused == 0) {
                break;
            }
        }

        // pick the best survivors, ignoring if they were picked in random
        for (unsigned int i = 0; i < std::min(scored_candidates.size(), num_best_survivors); i++) {
            survivors.push_back(scored_candidates[i].second);
        }

        // fill the rest of the survivors with reseeds
        for (unsigned int i = 0; i < final_population - survivors.size(); i++) {
            survivors.push_back(start_candidate);
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

    auto candidates = std::vector<Candidate>({start_candidate});
    Candidate the_winning_candidate;
    float the_winning_score = std::numeric_limits<float>::min();

    bool terminate = false;
    unsigned int num_iterations = 0;
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
                std::cout << " set new winning candidate! Candidate score: " << the_winning_score << "\n";
                std::cout << candidate << std::endl;
            }
        }
        // reduce the number of candidates
        candidates = culling_function(successors);

        num_iterations++;
        if (num_iterations > planning_args.max_iterations) {
            terminate = true;
        }

    } 

    // find the best candidate
    return the_winning_candidate.path_history;
}

std::ostream& operator<<(std::ostream& os, const Candidate& candidate) {
    os << "Candidate{"
       << "belief_trace=" << candidate.belief.cov_in_robot.trace() 
       << ", path_history=[";
    for (const auto& node : candidate.path_history) {
        os << node << ", ";
    }
    os << "]}";
    return os;
}

}  // namespace robot::experimental::beacon_sim
