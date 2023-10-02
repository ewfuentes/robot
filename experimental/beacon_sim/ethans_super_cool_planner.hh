

#include "planning/probabilistic_road_map.hh"
#include "planning/breadth_first_search.hh"
#include "experimental/beacon_sim/robot_belief.hh"
#include "experimental/beacon_sim/make_belief_updater.hh"


/*
BFS: given graph, search for paths 
BeliefRoadMap:  given graph, beacons, do belief updates
BeliefRoadMapPlanner: using these belief updates and search, find a nice path 

*/

namespace robot::experimental::beacon_sim {

constexpr int START_NODE_INDEX = -1;
constexpr int GOAL_NODE_INDEX = -2;

struct Candidate {
    RobotBelief belief; // where the robot thinks it is 
    std::vector<int> path_history; // the path that got it here
};

struct RollOutArgs {
    int num_roll_outs;

};


std::vector<Candidate> rollout ( const planning::RoadMap& map, 
                                 const Candidate& candidate, 
                                 const std::function(std::vector<planning::Successor<int>>(planning::Successor<int>)>& successor_function,
                                 const planning::BeliefUpdater<RobotBelief>& belief_updater,
                                 const RollOutArgs& roll_out_args );


struct CullingArgs {
    int num_candidates; // TODO: merge this with plannings args

};
std::vector<Candidate> cull_the_heard( const std::vector<Candidate>& candidates,
                                       const std::function<double(const Candidate&)>& scoring_function,
                                       const CullingArgs& cullingArgs );

std::function<double(const Candidate&)> make_scoring_function( const planning::RoadMap& map,
                                                               const Eigen::Vector2d& goal_state,
                                                               const planning::BeliefUpdater<RobotBelief>& belief_updater,
                                                               const std::vector<int>& beacons );


struct PlanningArgs {
    int num_candidates;
};
std::vector<int> plan( const planning::RoadMap& map, 
                       const Candidate& start_state,
                       const Eigen::Vector2d& goal_state,
                       const planning::BeliefUpdater<RobotBelief>& belief_updater,
                       const std::function<double(const Candidate&)>& scoring_function,
                       const std::function<std::vector<Candidate>(const planning::RoadMap&, const Candidate&, const planning::BeliefUpdater<RobotBelief>&, const RollOutArgs&)>& rollout_function,
                       const std::function<std::vector<Candidate>(const std::vector<Candidate>&, const std::function<double(const Candidate&)>&, const CullingArgs&)>& culling_function,
                       const PlanningArgs& planning_args );

} 