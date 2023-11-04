

#include "experimental/beacon_sim/make_belief_updater.hh"
#include "experimental/beacon_sim/robot_belief.hh"
#include "planning/breadth_first_search.hh"
#include "planning/probabilistic_road_map.hh"

#include <iostream>

/*
BFS: given graph, search for paths
BeliefRoadMap:  given graph, beacons, do belief updates
BeliefRoadMapPlanner: using these belief updates and search, find a nice path

*/

namespace robot::experimental::beacon_sim {

struct Candidate {
    RobotBelief belief;             // where the robot thinks it is
    std::vector<int> path_history;  // the path that got it here
};


struct RollOutArgs {
    unsigned int num_roll_outs;
};

using RolloutFunctionType = std::function<std::vector<Candidate>(const Candidate&)>;
RolloutFunctionType make_rollout_function ( const std::function<Candidate(const Candidate&)>& step_candidate,
                                            const std::function<bool(const Candidate&, int)>& terminate_rollout,
                                            const RollOutArgs& roll_out_args );

struct CullingArgs {
   unsigned long max_num_survivors;
   float entropy_proxy; // Number between 0 (only the best survive) and 1 (compleatly random selection of survivors)
   float reseed_percentage; // Number between 0 (no reseeding) and 1 (all new candidates from start)
};

using CullingFunctionType = std::function<std::vector<Candidate>(const std::vector<Candidate>&)>;
CullingFunctionType make_cull_the_heard_function( const std::function<double(const Candidate&)>& scoring_function, // if scoring function negative, kill the candidate
                                                  const Candidate& start_candidate,
                                                  const CullingArgs& cullingArgs );


struct PlanningArgs {
   unsigned int max_iterations;
};

// If positive, valid solution found
using GoalScoreFunctionType = std::function<double(const Candidate&)>;
std::vector<int> plan( const Candidate& start_candidate,
                      const RolloutFunctionType& rollout_function,
                      const CullingFunctionType& culling_function,
                      const GoalScoreFunctionType& goal_score_function,
                      const PlanningArgs& planning_args );

}  // namespace robot::experimental::beacon_sim
