
#include "experimental/beacon_sim/tick_sim.hh"

#include "common/liegroups/se2_to_proto.hh"
#include "common/time/robot_time.hh"
#include "common/time/robot_time_to_proto.hh"
#include "experimental/beacon_sim/beacon_observation_to_proto.hh"
#include "experimental/beacon_sim/ekf_slam_estimate_to_proto.hh"
#include "experimental/beacon_sim/generate_observations.hh"
#include "experimental/beacon_sim/information_lower_bound_planner.hh"
#include "experimental/beacon_sim/information_lower_bound_search.hh"
#include "experimental/beacon_sim/sim_config.hh"

namespace robot::experimental::beacon_sim {
namespace {
std::optional<RobotCommand> compute_command_from_plan(const time::RobotTimestamp &time_of_validity,
                                                      const liegroups::SE2 &local_from_est_robot,
                                                      const Plan &plan,
                                                      const time::RobotTimestamp::duration &dt) {
    constexpr double MAX_SPEED_MPS = 5.0;
    const auto time_since_plan_tov = time_of_validity - plan.time_of_validity;
    const std::vector<RobotBelief> &beliefs = plan.brm_plan.beliefs;
    time::RobotTimestamp::duration time_along_plan(0);
    for (int i = 1; i < static_cast<int>(beliefs.size()); i++) {
        const Eigen::Vector2d prev_node_in_local = beliefs.at(i - 1).local_from_robot.translation();
        const Eigen::Vector2d next_node_in_local = beliefs.at(i).local_from_robot.translation();
        const Eigen::Vector2d next_in_prev = next_node_in_local - prev_node_in_local;
        const double dist_m = next_in_prev.norm();
        const time::RobotTimestamp::duration step_dt = time::as_duration(dist_m / MAX_SPEED_MPS);
        if (time_along_plan + step_dt > time_since_plan_tov) {
            // Compute expected pose
            // Construct a pose at the start node facing the next node
            const double heading_rad = std::atan2(next_in_prev.y(), next_in_prev.x());

            const liegroups::SE2 local_from_start_node =
                liegroups::SE2(liegroups::SO2(heading_rad), prev_node_in_local);
            const double frac =
                (time_since_plan_tov - time_along_plan) / std::chrono::duration<double>(step_dt);
            const liegroups::SE2 start_from_expected = liegroups::SE2::trans(frac * dist_m, 0.0);
            const liegroups::SE2 local_from_expected = local_from_start_node * start_from_expected;

            // Compute command
            const Eigen::Vector2d expected_in_estimated =
                local_from_est_robot.inverse() * local_from_expected.translation();
            const Eigen::Vector2d target_in_estimated =
                local_from_est_robot.inverse() * next_node_in_local;

            RobotCommand command;
            command.turn_rad = std::atan2(target_in_estimated.y(), target_in_estimated.x());
            command.move_m = std::min(MAX_SPEED_MPS * std::chrono::duration<double>(dt).count(),
                                      expected_in_estimated.norm());
            return command;
        } else {
            time_along_plan += step_dt;
        }
    }
    return std::nullopt;
}

std::optional<planning::BRMPlan<RobotBelief>> run_brm_planner(const BeaconSimState &state,
                                                              const bool allow_brm_backtracking,
                                                              const ObservationConfig &obs_config) {
    constexpr double UNCERTAINTY_TOLERANCE = 0.1;
    std::cout << "Starting to Plan" << std::endl;
    const BeliefRoadMapOptions options = {
        .max_sensor_range_m = obs_config.max_sensor_range_m.value(),
        .uncertainty_tolerance =
            allow_brm_backtracking ? std::make_optional(UNCERTAINTY_TOLERANCE) : std::nullopt,
        .max_num_edge_transforms = 1000,
    };
    const auto brm_plan = compute_belief_road_map_plan(state.road_map, state.ekf,
                                                       state.map.beacon_potential(), options);
    std::cout << "plan complete" << std::endl;
    for (int idx = 0; idx < static_cast<int>(brm_plan->nodes.size()); idx++) {
        std::cout << idx << " " << brm_plan->nodes.at(idx) << " "
                  << brm_plan->beliefs.at(idx).local_from_robot.translation().transpose()
                  << " cov det: " << brm_plan->beliefs.at(idx).cov_in_robot.determinant()
                  << std::endl;
    }
    return brm_plan;
}

std::optional<planning::BRMPlan<RobotBelief>> run_info_lower_bound_planner(
    const BeaconSimState &state, const double max_sensor_range_m,
    const double info_lower_bound_at_goal) {
    return compute_info_lower_bound_plan(state.road_map, state.ekf, info_lower_bound_at_goal,
                                         max_sensor_range_m);
}

}  // namespace

proto::BeaconSimDebug tick_sim(const SimConfig &config, const RobotCommand &command,
                               InOut<BeaconSimState> state) {
    constexpr ObservationConfig OBS_CONFIG = {
        .range_noise_std_m = 0.1,
        .max_sensor_range_m = 5.0,
    };

    RobotCommand next_command = command;
    if (config.enable_brm_planner || config.enable_info_lower_bound_planner) {
        // Plan
        const bool have_plan = state->plan.has_value();
        const bool have_goal = state->goal.has_value();
        const bool should_plan =
            have_goal && (!have_plan || (have_plan && state->goal->time_of_validity >
                                                          state->plan->time_of_validity));
        if (should_plan) {
            state->road_map.add_start_goal(
                {.start = state->ekf.estimate().local_from_robot().translation(),
                 .goal = state->goal->goal_position,
                 .connection_radius_m = 5.0});
            if (config.enable_brm_planner) {
                const auto maybe_plan =
                    run_brm_planner(*state, config.allow_brm_backtracking, OBS_CONFIG);
                if (maybe_plan.has_value()) {
                    state->plan = {.time_of_validity = state->time_of_validity,
                                   .brm_plan = maybe_plan.value()};
                }
            } else if (config.enable_info_lower_bound_planner) {
                const auto maybe_plan =
                    run_info_lower_bound_planner(*state, OBS_CONFIG.max_sensor_range_m.value(),
                                                 config.info_lower_bound_at_goal.value());
                if (maybe_plan.has_value()) {
                    state->plan = {.time_of_validity = state->time_of_validity,
                                   .brm_plan = maybe_plan.value()};
                } else {
                    std::cout << "Infeasible goal lower bound constraint" << std::endl;
                }
            }
        }
        if (state->plan.has_value()) {
            // Figure out which which node we should be targeting
            const auto plan_command = compute_command_from_plan(
                state->time_of_validity, state->ekf.estimate().local_from_robot(),
                state->plan.value(), config.dt);
            if (plan_command.has_value()) {
                next_command.turn_rad = plan_command->turn_rad;
                next_command.move_m = plan_command->move_m;
            }
        }
    }
    // simulate robot forward
    state->robot.turn(next_command.turn_rad);
    state->robot.move(next_command.move_m);

    state->map.update(state->time_of_validity);

    proto::BeaconSimDebug debug_msg;
    pack_into(state->time_of_validity, debug_msg.mutable_time_of_validity());
    pack_into(state->ekf.estimate(), debug_msg.mutable_prior());

    const liegroups::SE2 old_robot_from_new_robot = liegroups::SE2::rot(next_command.turn_rad) *
                                                    liegroups::SE2::trans(next_command.move_m, 0.0);
    pack_into(old_robot_from_new_robot, debug_msg.mutable_old_robot_from_new_robot());

    pack_into(state->ekf.predict(state->time_of_validity, old_robot_from_new_robot),
              debug_msg.mutable_prediction());

    // generate observations
    state->observations = generate_observations(state->time_of_validity, state->map, state->robot,
                                                OBS_CONFIG, make_in_out(state->gen));
    pack_into(state->observations, debug_msg.mutable_observations());

    const auto &ekf_estimate = state->ekf.update(state->observations);
    pack_into(ekf_estimate, debug_msg.mutable_posterior());
    pack_into(state->robot.local_from_robot(), debug_msg.mutable_local_from_true_robot());

    return debug_msg;
}

}  // namespace robot::experimental::beacon_sim
