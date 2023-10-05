#include "experimental/beacon_sim/make_eigen_bounds.hh"
#include <iostream>


namespace robot::experimental::beacon_sim {

double compute_backwards_edge_belief_transform(
    const liegroups::SE2 &local_from_robot, // end state transform
    const Eigen::Vector2d &start_state_in_local, // start state in local
    const EkfSlamConfig &ekf_config, const EkfSlamEstimate &ekf_estimate,
    const std::optional<std::vector<int>> &available_beacons,
    const double max_sensor_range_m,
    const TransformType type) {
    constexpr double DT_S = 0.5;
    constexpr double VELOCITY_MPS = 2.0;
    constexpr double ANGULAR_VELOCITY_RADPS = 2.0;

    liegroups::SE2 local_from_new_robot = local_from_robot;
    constexpr double TOL = 1e-6;
    for (Eigen::Vector2d start_in_robot = local_from_robot.inverse() * start_state_in_local;
         start_in_robot.norm() > TOL;
         start_in_robot = local_from_new_robot.inverse() * start_state_in_local) {
        // Move towards the goal
        const liegroups::SE2 old_robot_from_new_robot = [&]() {
            const double angle_to_goal_rad = std::atan2(start_in_robot.y(), start_in_robot.x());

            if (std::abs(angle_to_goal_rad) > TOL) {
                // First turn to face the goal
                constexpr double MAX_ANGLE_STEP_RAD = DT_S * ANGULAR_VELOCITY_RADPS;
                const double angle_step_rad = std::copysign(
                    std::min(std::abs(angle_to_goal_rad), MAX_ANGLE_STEP_RAD), angle_to_goal_rad);
                return liegroups::SE2::rot(angle_step_rad);
            } else {
                // Then drive towards the goal
                constexpr double MAX_DIST_STEP_M = DT_S * VELOCITY_MPS;
                const double dist_step_m = std::min(start_in_robot.x(), MAX_DIST_STEP_M);
                return liegroups::SE2::transX(dist_step_m);
            }
        }();

        // Update the mean
        local_from_new_robot = local_from_new_robot * old_robot_from_new_robot;

        // Compute the process update for the covariance
        const Eigen::Matrix3d process_noise_in_robot =
            compute_process_noise(ekf_config, DT_S, old_robot_from_new_robot.arclength());

        std::cout << "produced process noise " << process_noise_in_robot << std::endl;

        const TypedTransform process_transform =
            compute_process_transform(process_noise_in_robot, old_robot_from_new_robot, type);

        std::cout << "produced process transform " << process_transform << std::endl;

        // Compute the measurement update for the covariance
        const TypedTransform measurement_transform =
            compute_measurement_transform(local_from_new_robot, ekf_config, ekf_estimate,
                                          available_beacons, max_sensor_range_m, type);
        std::cout << "produced measurement transform " << measurement_transform << std::endl;

        // scattering_transform = (scattering_transform * process_transform).value();
        // scattering_transform = (scattering_transform * measurement_transform).value();
    }

    return std::make_tuple(liegroups::SE2(local_from_new_robot.so2().log(), end_state_in_local),
                           scattering_transform);
}


} //namespace 