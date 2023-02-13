
#include "experimental/beacon_sim/tick_sim.hh"

#include "common/liegroups/se2_to_proto.hh"
#include "common/time/robot_time.hh"
#include "common/time/robot_time_to_proto.hh"
#include "experimental/beacon_sim/beacon_observation_to_proto.hh"
#include "experimental/beacon_sim/ekf_slam_estimate_to_proto.hh"

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
}  // namespace

proto::BeaconSimDebug tick_sim(const SimConfig &config, const RobotCommand &command,
                               InOut<BeaconSimState> state) {
    constexpr ObservationConfig OBS_CONFIG = {
        .range_noise_std_m = 0.1,
        .max_sensor_range_m = 5.0,
    };

    RobotCommand next_command = command;
    if (config.enable_brm_planner) {
        // Plan
        const bool have_plan = state->plan.has_value();
        const bool have_goal = state->goal.has_value();
        const bool should_plan =
            have_goal && (!have_plan || (have_plan && state->goal->time_of_validity >
                                                          state->plan->time_of_validity));
        if (should_plan) {
            constexpr int NUM_START_CONNECTIONS = 6;
            constexpr int NUM_GOAL_CONNECTIONS = 6;
            constexpr double UNCERTAINTY_TOLERANCE = 0.1;
            std::cout << "Starting to Plan" << std::endl;
            const auto brm_plan = compute_belief_road_map_plan(
                state->road_map, state->ekf, state->goal->goal_position,
                OBS_CONFIG.max_sensor_range_m.value(), NUM_START_CONNECTIONS, NUM_GOAL_CONNECTIONS,
                UNCERTAINTY_TOLERANCE);
            std::cout << "plan complete" << std::endl;
            for (int idx = 0; idx < static_cast<int>(brm_plan->nodes.size()); idx++) {
                std::cout << idx << " " << brm_plan->nodes.at(idx) << " "
                          << brm_plan->beliefs.at(idx).local_from_robot.translation().transpose()
                          << " cov det: " << brm_plan->beliefs.at(idx).cov_in_robot.determinant()
                          << std::endl;
            }
            state->plan = {.time_of_validity = state->time_of_validity,
                           .brm_plan = brm_plan.value()};
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
