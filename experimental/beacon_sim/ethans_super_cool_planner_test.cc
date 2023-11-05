#include "experimental/beacon_sim/ethans_super_cool_planner.hh"

#include "experimental/beacon_sim/test_helpers.hh"
#include "gtest/gtest.h"
#include "planning/probabilistic_road_map.hh"
#include "common/check.hh"

#include <iostream>

namespace robot::experimental::beacon_sim {

using StepCandidateFunc = std::function<Candidate(const Candidate&)>;

StepCandidateFunc make_random_step_candidate_function(const planning::RoadMap &map, const planning::BeliefUpdater<RobotBelief> &belief_updater) {
    return [&map, &belief_updater](const Candidate& parent)->Candidate {
        std::vector<planning::Successor<int>> successors;
        for (const auto &[neighbor_idx, neighbor_pos] : map.neighbors(parent.path_history.back())) {
            successors.push_back({
                .state = neighbor_idx,
                .edge_cost = (neighbor_pos - map.point(parent.path_history.back())).norm(),
            });
        }
        int action_index = rand() % successors.size();
        // traverse the edge
        Candidate child = parent;
        child.belief = belief_updater(child.belief, child.path_history.back(), successors[action_index].state);
        child.path_history.push_back(successors[action_index].state);

        return child;
    };
}

using TerminateRolloutFunc = std::function<bool(const Candidate &, int)>;

TerminateRolloutFunc make_max_steps_or_goal_terminate_func(const int max_steps, const planning::RoadMap& map) {
    return [max_steps, &map](const Candidate &candidate, const int num_steps) {
        // if at goal, terminate rollout 
        if (candidate.path_history.back() == map.GOAL_IDX) {
            return true;
        }
        return num_steps >= max_steps;
    };
}


TEST(EthansSuperCoolPlannerTest, RolloutHappyCase) {
    // Setup

    const EkfSlamConfig ekf_config{
        .max_num_beacons = 1,
        .initial_beacon_uncertainty_m = 100.0,
        .along_track_process_noise_m_per_rt_meter = 0.05,
        .cross_track_process_noise_m_per_rt_meter = 0.05,
        .pos_process_noise_m_per_rt_s = 0.0,
        .heading_process_noise_rad_per_rt_meter = 1e-3,
        .heading_process_noise_rad_per_rt_s = 0.0,
        .beacon_pos_process_noise_m_per_rt_s = 1e-6,
        .range_measurement_noise_m = 1e-1,
        .bearing_measurement_noise_rad = 1e-1,
        .on_map_load_position_uncertainty_m = 2.0,
        .on_map_load_heading_uncertainty_rad = 0.5,
    };

    const auto &[road_map, ekf_slam, potential] = create_grid_environment(ekf_config, 0.5);


    Candidate candidate = {
        .belief = ekf_slam.estimate().robot_belief(),
        .path_history = {road_map.START_IDX},
    };

    RollOutArgs roll_out_args = {
        .num_roll_outs = 1000,
    };

    constexpr double max_sensor_range_m = 3.0;

    planning::BeliefUpdater<RobotBelief> belief_updater =
        make_belief_updater(road_map, 
                            max_sensor_range_m,
                            ekf_slam,
                            std::nullopt,
                            TransformType::INFORMATION);

    const StepCandidateFunc step_func = make_random_step_candidate_function(road_map, belief_updater);
    const TerminateRolloutFunc terminate_rollout_func = make_max_steps_or_goal_terminate_func(10, road_map);

    // Action

    const RolloutFunctionType rollout_function = make_rollout_function(step_func, terminate_rollout_func, roll_out_args);

    auto candidates = rollout_function(candidate);

    std::cout << "from initial candidate: " << candidate << std::endl;
    for (const auto &candidate : candidates) {
        std::cout << "\t candidate: " << candidate << std::endl;
    }
    // Verification
    EXPECT_TRUE(candidates.size() == roll_out_args.num_roll_outs);
}

using ScoringFunc = std::function<double(const Candidate&)>;
/*
    If the scoring function is negative, the candidate is killed
    If the scoring function is positive, the candidate is kept, larger scores are better solutions
*/
ScoringFunc make_goal_distance_scoring_function( const planning::RoadMap& map, 
                                                 const unsigned long max_path_steps, 
                                                 const double max_dist_from_goal_m ) {
    return [&map, max_path_steps, max_dist_from_goal_m](const Candidate& candidate) {
        // if the candidate is over the max path length, return -inf
        if (candidate.path_history.size() > max_path_steps) {
            return -std::numeric_limits<double>::infinity();
        }
        // 0 <= length_reward <= 1, larger means shorter path
        double length_reward = (max_path_steps - candidate.path_history.size()) / max_path_steps;
        // 0 <= goal_distance_reward <= 1, larger means closer to goal. MAP DEPENDENT
        double goal_distance_reward = 1 - (max_dist_from_goal_m-(map.point(candidate.path_history.back()) - map.point(map.GOAL_IDX)).norm()) / max_dist_from_goal_m;
        return 0.5 * length_reward + 0.5 * goal_distance_reward;
    };
}
/*
    If the goal scoring function is negative, the path does not satisfy the constraints
    If the goal scoring function is positive, the path satisfies the constraints, larger scores are better solutions
*/
GoalScoreFunctionType make_goal_scoring_function(const planning::RoadMap& map,
                                                 const double max_trace
                                                ) 
{
    return [&map, max_trace](const Candidate& candidate){
        // if the candidate is not at the goal, or the covariance is too high, return -inf  
        if (candidate.path_history.back() != map.GOAL_IDX || candidate.belief.cov_in_robot.trace() > max_trace) {
            return -std::numeric_limits<double>::infinity();
        }
        
        // if the path length is acceptreturn 1 / trace(covariance)
        return 1 / candidate.belief.cov_in_robot.trace();

    };

}

TEST(EthansSuperCoolPlannerTest, CullingHappyCase) {
    // Setup

    const EkfSlamConfig ekf_config{
        .max_num_beacons = 1,
        .initial_beacon_uncertainty_m = 100.0,
        .along_track_process_noise_m_per_rt_meter = 0.05,
        .cross_track_process_noise_m_per_rt_meter = 0.05,
        .pos_process_noise_m_per_rt_s = 0.0,
        .heading_process_noise_rad_per_rt_meter = 1e-3,
        .heading_process_noise_rad_per_rt_s = 0.0,
        .beacon_pos_process_noise_m_per_rt_s = 1e-6,
        .range_measurement_noise_m = 1e-1,
        .bearing_measurement_noise_rad = 1e-1,
        .on_map_load_position_uncertainty_m = 2.0,
        .on_map_load_heading_uncertainty_rad = 0.5,
    };

    const auto &[road_map, ekf_slam, potential] = create_grid_environment(ekf_config, 0.5);


    Candidate candidate = {
        .belief = ekf_slam.estimate().robot_belief(),
        .path_history = {road_map.START_IDX},
    };

    RollOutArgs roll_out_args = {
        .num_roll_outs = 1000,
    };

    constexpr double max_sensor_range_m = 3.0;

    planning::BeliefUpdater<RobotBelief> belief_updater =
        make_belief_updater(road_map, 
                            max_sensor_range_m,
                            ekf_slam,
                            std::nullopt,
                            TransformType::INFORMATION);

    const StepCandidateFunc step_func = make_random_step_candidate_function(road_map, belief_updater);
    const TerminateRolloutFunc terminate_rollout_func = make_max_steps_or_goal_terminate_func(10, road_map);
    const RolloutFunctionType rollout_function = make_rollout_function(step_func, terminate_rollout_func, roll_out_args);
    auto candidates = rollout_function(candidate);

    const ScoringFunc scoring_func = make_goal_distance_scoring_function(road_map, 10, 20);

    CullingArgs culling_args = {
        .max_num_survivors = 100,
        .entropy_proxy = 0.2,
    };

    // Action
    const CullingFunctionType culling_function = make_cull_the_heard_function(scoring_func, candidate, culling_args);
    auto culled_candidates = culling_function(candidates);


    std::cout << "Culled candidates to: " << std::endl;
    for (const auto &candidate : culled_candidates) {
        std::cout << "\t candidate: " << candidate << " with score " << scoring_func(candidate) << std::endl;
    }
    // Verification
    EXPECT_TRUE(culled_candidates.size() == culling_args.max_num_survivors);

    // Check that at least 1 - entropy_proxy % of the survivors have the top scores
    // It's at least because the random sampling happens first and could have picked the best 

    std::vector<std::pair<double, Candidate>> scored_candidates;
    for (const auto& candidate : candidates) {
        scored_candidates.push_back({scoring_func(candidate), candidate});
    }
    std::sort(scored_candidates.begin(), scored_candidates.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    unsigned int num_top_candidates = culling_args.max_num_survivors * (1.0 - culling_args.entropy_proxy);
    unsigned int num_top_candidates_in_culled = 0;
    double score_to_beat = scored_candidates[num_top_candidates - 1].first;
    for (const auto& candidate : culled_candidates) {
        if (scoring_func(candidate) >= score_to_beat) {
            num_top_candidates_in_culled++;
        }
    }
    EXPECT_TRUE(num_top_candidates_in_culled >= num_top_candidates);
}


TEST(EthansSuperCoolPlannerTest, PuttingItAllTogether) {
    // Setup

    const EkfSlamConfig ekf_config{
        .max_num_beacons = 1,
        .initial_beacon_uncertainty_m = 100.0,
        .along_track_process_noise_m_per_rt_meter = 0.05,
        .cross_track_process_noise_m_per_rt_meter = 0.05,
        .pos_process_noise_m_per_rt_s = 0.0,
        .heading_process_noise_rad_per_rt_meter = 1e-3,
        .heading_process_noise_rad_per_rt_s = 0.0,
        .beacon_pos_process_noise_m_per_rt_s = 1e-6,
        .range_measurement_noise_m = 1e-1,
        .bearing_measurement_noise_rad = 1e-1,
        .on_map_load_position_uncertainty_m = 2.0,
        .on_map_load_heading_uncertainty_rad = 0.5,
    };

    const auto &[road_map, ekf_slam, potential] = create_grid_environment(ekf_config, 0.5);


    Candidate candidate = {
        .belief = ekf_slam.estimate().robot_belief(),
        .path_history = {road_map.START_IDX},
    };

    RollOutArgs roll_out_args = {
        .num_roll_outs = 1000,
    };

    constexpr double max_sensor_range_m = 3.0;

    planning::BeliefUpdater<RobotBelief> belief_updater =
        make_belief_updater(road_map, 
                            max_sensor_range_m,
                            ekf_slam,
                            std::nullopt,
                            TransformType::INFORMATION);

    const StepCandidateFunc step_func = make_random_step_candidate_function(road_map, belief_updater);
    const TerminateRolloutFunc terminate_rollout_func = make_max_steps_or_goal_terminate_func(10, road_map);
    const RolloutFunctionType rollout_function = make_rollout_function(step_func, terminate_rollout_func, roll_out_args);

    const ScoringFunc scoring_func = make_goal_distance_scoring_function(road_map, 10, 20);

    CullingArgs culling_args = {
        .max_num_survivors = 100,
        .entropy_proxy = 0.2,
    };

    const CullingFunctionType culling_function = make_cull_the_heard_function(scoring_func, candidate, culling_args);

    const GoalScoreFunctionType goal_score_function = make_goal_scoring_function(road_map, 100);

    // Action

    const PlanningArgs planning_args = {
        .max_iterations = 100,
    };

    std::vector<int> resulting_path = plan(candidate, rollout_function, culling_function, goal_score_function, planning_args);

    std::cout << "Resulting path: " << std::endl;
    for (const auto &node : resulting_path) {
        std::cout << "\t node: " << node << std::endl;
    }


}

}  // namespace robot::experimental::beacon_sim