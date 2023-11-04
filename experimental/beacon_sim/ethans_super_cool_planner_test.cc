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

TerminateRolloutFunc make_terminate_func(const int max_steps) {
    return [max_steps]([[maybe_unused]] const Candidate &candidate, const int num_steps) {
        return num_steps >= max_steps;
    };
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
    const TerminateRolloutFunc terminate_rollout_func = make_terminate_func(10);

    // Action

    auto candidates = rollout(step_func, 
                              terminate_rollout_func, 
                              candidate,
                              roll_out_args);

    std::cout << "from initial candidate: " << candidate << std::endl;
    for (const auto &candidate : candidates) {
        std::cout << "\t candidate: " << candidate << std::endl;
    }
    // Verification
    EXPECT_TRUE(candidates.size() == roll_out_args.num_roll_outs);
}

using ScoringFunc = std::function<double(const Candidate&)>;
ScoringFunc make_goal_distance_scoring_function( const planning::RoadMap& map ) {
    return [&map](const Candidate& candidate) {
        return -(map.point(candidate.path_history.back()) - map.point(map.GOAL_IDX)).norm();
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

    const SuccessorFunc successor_func = make_successor_function(road_map);
    const TerminateRolloutFunc terminate_rollout_func = make_max_steps_terminate_func(10);

    auto candidates = rollout(road_map, 
                              terminate_rollout_func, 
                              candidate,
                              successor_func, 
                              belief_updater, 
                              roll_out_args);

    const ScoringFunc scoring_func = make_goal_distance_scoring_function(road_map);

    CullingArgs culling_args = {
        .num_survivors = 100,
        .entropy_proxy = 0.2,
    };

    // Action

    auto culled_candidates = cull_the_heard(candidates,
                                            scoring_func, 
                                            culling_args);


    std::cout << "Culled candidates to: " << std::endl;
    for (const auto &candidate : culled_candidates) {
        std::cout << "\t candidate: " << candidate << " with score " << scoring_func(candidate) << std::endl;
    }
    // Verification
    EXPECT_TRUE(culled_candidates.size() == culling_args.num_survivors);

    // Check that at least 1 - entropy_proxy % of the survivors have the top scores
    // It's at least because the random sampling happens first and could have picked the best 

    std::vector<std::pair<double, Candidate>> scored_candidates;
    for (const auto& candidate : candidates) {
        scored_candidates.push_back({scoring_func(candidate), candidate});
    }
    std::sort(scored_candidates.begin(), scored_candidates.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    unsigned int num_top_candidates = culling_args.num_survivors * (1.0 - culling_args.entropy_proxy);
    unsigned int num_top_candidates_in_culled = 0;
    double score_to_beat = scored_candidates[num_top_candidates - 1].first;
    for (const auto& candidate : culled_candidates) {
        if (scoring_func(candidate) >= score_to_beat) {
            num_top_candidates_in_culled++;
        }
    }
    EXPECT_TRUE(num_top_candidates_in_culled >= num_top_candidates);
}

}  // namespace robot::experimental::beacon_sim


}  // namespace robot::experimental::beacon_sim
