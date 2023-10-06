#include "experimental/beacon_sim/make_eigen_bounds.hh"
#include <iostream>


namespace robot::experimental::beacon_sim {


double step_lower_eigen_bound( // returns a lower bound on min ev of \Omega_{t-1}
    const double lower_eigen_value_information, // min ev of \Omega_t
    const double upper_eigen_value_dynamics, // max ev of G_t^{-1} G_t^{-T}
    const double upper_eigen_value_measurement, // max ev of M_t
    const double lower_eigen_value_process_noise // min ev of R_t
) {
    return 1. / (upper_eigen_value_dynamics * (1. / (lower_eigen_value_information - upper_eigen_value_measurement) - lower_eigen_value_process_noise));
}

/// Start state is the end of the edge
/// End state is the start of the edge (as we move backwards)
double compute_backwards_edge_belief_transform(
    const double lower_eigen_value_information, // min ev of \Omega_t
    const liegroups::SE2 &local_from_robot, // end node state transform
    const Eigen::Vector2d &end_state_in_local, // start node state in local
    const EkfSlamConfig &ekf_config, 
    const EkfSlamEstimate &ekf_estimate,
    const std::optional<std::vector<int>> &available_beacons,
    const double max_sensor_range_m) {
    constexpr double DT_S = 0.5;
    constexpr double VELOCITY_MPS = 2.0;
    constexpr double ANGULAR_VELOCITY_RADPS = 2.0;

    liegroups::SE2 local_from_new_robot = local_from_robot;
    constexpr double TOL = 1e-6;

    double information_lower_bound = lower_eigen_value_information;
    for (Eigen::Vector2d start_in_robot = local_from_robot.inverse() * end_state_in_local;
         start_in_robot.norm() > TOL;
         start_in_robot = local_from_new_robot.inverse() * end_state_in_local) {
        // Move towards the goal
        const liegroups::SE2 old_robot_from_new_robot = [&]() {
            const double angle_to_goal_rad = std::atan2(start_in_robot.y(), -1.0 * start_in_robot.x()); //angle is 0 when goal is negative x axis

            if (std::abs(angle_to_goal_rad) > TOL) {
                // First turn until our back faces the goal
                constexpr double MAX_ANGLE_STEP_RAD = DT_S * ANGULAR_VELOCITY_RADPS;
                const double angle_step_rad = -1 * std::copysign(
                    std::min(std::abs(angle_to_goal_rad), MAX_ANGLE_STEP_RAD), angle_to_goal_rad);
                return liegroups::SE2::rot(angle_step_rad);
            } else {
                // Then drive backwards towards the goal
                constexpr double MAX_DIST_STEP_M = DT_S * VELOCITY_MPS;
                const double dist_step_m = -1.0 * std::min(std::abs(start_in_robot.x()), MAX_DIST_STEP_M);
                return liegroups::SE2::transX(dist_step_m);
            }
        }();

        // Update the mean
        local_from_new_robot = local_from_new_robot * old_robot_from_new_robot;

        // G_t^{-1}
        const Eigen::Matrix3d inv_dynamics_jac_wrt_state = old_robot_from_new_robot.Adj();
        // max_eigen_value(G_t^{-1} G_t^{-T})
        const double dynamics_upper_eigen_value_bound = (inv_dynamics_jac_wrt_state * inv_dynamics_jac_wrt_state.transpose()).eigenvalues().real().maxCoeff();
        std::cout << "inv_dynamics_jac_wrt_state: " << std::endl << inv_dynamics_jac_wrt_state << std::endl;
        std::cout << "dynamics_upper_eigen_value_bound: " << dynamics_upper_eigen_value_bound << std::endl;
        // R_t
        const Eigen::Matrix3d process_noise = compute_process_noise(ekf_config, DT_S, old_robot_from_new_robot.arclength());
        const double process_lower_eigen_value_bound = process_noise.eigenvalues().real().minCoeff();
        std::cout << "process_noise: " << std::endl << process_noise << std::endl;
        std::cout << "process_lower_eigen_value_bound: " << process_lower_eigen_value_bound << std::endl;

        // M_t
        TransformType type = TransformType::COVARIANCE;
        const TypedTransform measurement_transform =
            compute_measurement_transform(local_from_new_robot, ekf_config, ekf_estimate,
                                          available_beacons, max_sensor_range_m, type);
        const Eigen::Matrix3d M_t = -1 * std::get<ScatteringTransform<TransformType::COVARIANCE>>(measurement_transform).bottomLeftCorner(3,3);
        const double measurement_upper_eigen_value_bound = M_t.eigenvalues().real().maxCoeff();
        std::cout << "M_t: " << std::endl << M_t << std::endl;
        std::cout << "measurement_upper_eigen_value_bound: " << measurement_upper_eigen_value_bound << std::endl;

        

        // scattering_transform = (scattering_transform * process_transform).value();
        // scattering_transform = (scattering_transform * measurement_transform).value();
        information_lower_bound = step_lower_eigen_bound(
            information_lower_bound,
            dynamics_upper_eigen_value_bound,
            measurement_upper_eigen_value_bound,
            process_lower_eigen_value_bound
        );
        std::cout << "information_lower_bound: " << information_lower_bound << std::endl;
    }
    return information_lower_bound;
}


} //namespace 