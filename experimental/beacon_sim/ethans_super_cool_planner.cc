#include "experimental/beacon_sim/ethans_super_cool_planner.hh" 


namespace robot::experimental::beacon_sim {



std::vector<Candidate> rollout ( [[maybe_unused]] const planning::RoadMap& map, 
                                 std::function<bool(const Candidate&, int)> terminate_rollout,
                                 [[maybe_unused]] const Candidate& candidate, 
                                 [[maybe_unused]] const planning::BeliefUpdater<RobotBelief>& belief_updater,
                                 [[maybe_unused]] const RollOutArgs& roll_out_args ){
    // what node are we on

    std::vector<Candidate> successors;
    int position = ...;

    for (int rollout_i = 0; rollout_i < roll_out_args.num_roll_outs; rollout_i++) {
        int num_steps = 0;
        Candidate start = candidate;
        while(not terminate_rollout(start, num_steps)){
            // pick an action

            // traverse the edge


            num_steps++;
        }

        successors.push_back(start);
    }

    return successors;
}

std::vector<Candidate> cull_the_heard( const std::vector<Candidate>& candidates,
                                       [[maybe_unused]] const std::function<double(const Candidate&)>& scoring_function,
                                       const CullingArgs& cullingArgs ) {

    // pick randomly from the candidates, without replacement 
    std::vector<Candidate> survivors;
    std::vector<bool> used(candidates.size(), false);
    for (int i = 0; i < cullingArgs.num_candidates; i++){
        int index = rand() % candidates.size();
        while (used[index]){
            index = rand() % candidates.size();
        }
        used[index] = true;
        survivors.push_back(candidates[index]);
    }

    return survivors;
}

std::function<double(const Candidate&)> make_scoring_function( const planning::RoadMap& map,
                                                               const Eigen::Vector2d& goal_state,
                                                               const planning::BeliefUpdater<RobotBelief>& belief_updater,
                                                               const std::vector<int>& beacons ) {
    return [map, goal_state, belief_updater, beacons](const Candidate& candidate){
        // TODO
        return 0.0;
    };

}


std::vector<int> plan( const planning::RoadMap& map, 
                       const Candidate& start_state,
                       const Eigen::Vector2d& goal_state,
                       const planning::BeliefUpdater<RobotBelief>& belief_updater,
                       const std::function<double(const Candidate&)>& scoring_function,
                       const std::function<std::vector<Candidate>(const planning::RoadMap&, const Candidate&, const planning::BeliefUpdater<RobotBelief>&, const RollOutArgs&)>& rollout_function,
                       const std::function<std::vector<Candidate>(const std::vector<Candidate>&, const std::function<double(const Candidate&)>&, const CullingArgs&)>& culling_function,
                       const PlanningArgs& planning_args ) {

    auto candidates = std::vector<Candidate>(start_state, planning_args.num_candidates);

    std::vector<Candidate> successful_candidates; // list of candidates that have reached the goal
    
    while (not terminate):
        std::vector<Candidate> successors;
        for (const auto& candidate : candidates){
            successors.extend( rollout( map, candidate, rolloutArgs ) );
        }
        for (const auto& candidate : successors){
            // TODO
        }

        candidates = cull_the_heard( candidates, scoring_function, cullingArgs);

    // find the best candidate
    best_candidate = candidates[0];

    return best_candidate.path_history;

}

}
