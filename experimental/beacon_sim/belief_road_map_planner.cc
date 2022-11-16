
#include "experimental/beacon_sim/belief_road_map_planner.hh"

#include <iostream>

namespace robot::experimental::beacon_sim {
namespace {
planning::BeliefUpdater<RobotBelief> make_belief_updater(const planning::RoadMap &road_map,
                                                         const Eigen::Vector2d &goal_state,
                                                         const double max_sensor_range_m,
                                                         const EkfSlam &ekf) {
    return [&road_map, goal_state, max_sensor_range_m, &ekf](
               const RobotBelief &initial_belief, const int start_idx, const int end_idx) {
        std::cout << "Traversing " << start_idx << " to " << end_idx << std::endl;
        constexpr double DT_S = 0.1;
        constexpr double VELOCITY_MPS = 2.0;
        constexpr double ANGULAR_VELOCITY_RADPS = 2.0;
        const Eigen::Vector2d end_position =
            end_idx >= 0 ? road_map.points.at(end_idx) : goal_state;

        std::cout << "End Position: " << end_position.transpose() << std::endl;
        RobotBelief out = initial_belief;
        const EkfSlamConfig &config = ekf.config();
        const EkfSlamEstimate &est = ekf.estimate();

        while ((out.local_from_robot.inverse() * end_position).norm() > 1e-6) {
            // Prediction Update
            {
                // Compute the control action to get us closer to the goal
                const liegroups::SE2 old_robot_from_new_robot = [&]() {
                    constexpr double ANGLE_TOL = 1e-6;
                    const Eigen::Vector2d goal_in_robot =
                        out.local_from_robot.inverse() * goal_state;
                    const double angle_to_goal_rad =
                        std::atan2(goal_in_robot.y(), goal_in_robot.x());
                    if (std::abs(angle_to_goal_rad) > ANGLE_TOL) {
                        constexpr double MAX_STEP_TURN_RAD = ANGULAR_VELOCITY_RADPS * DT_S;
                        const double step_turn_amount_rad =
                            std::copysign(std::min(std::abs(angle_to_goal_rad), MAX_STEP_TURN_RAD),
                                          angle_to_goal_rad);
                        return liegroups::SE2::rot(step_turn_amount_rad);
                    } else {
                        constexpr double MAX_STEP_DIST_M = VELOCITY_MPS * DT_S;
                        const double dist_to_goal_m = goal_in_robot.norm();
                        const double step_travel_m = std::min(dist_to_goal_m, MAX_STEP_DIST_M);
                        return liegroups::SE2::transX(step_travel_m);
                    }
                }();

                const auto sq = [](const double x) { return x * x; };
                const Eigen::Matrix3d process_noise_in_robot = Eigen::DiagonalMatrix<double, 3>{
                    // Along track noise
                    sq(config.along_track_process_noise_m_per_rt_meter) *
                            old_robot_from_new_robot.arclength() +
                        sq(config.pos_process_noise_m_per_rt_s) * DT_S,
                    // Cross track noise
                    sq(config.cross_track_process_noise_m_per_rt_meter) *
                            old_robot_from_new_robot.arclength() +
                        sq(config.pos_process_noise_m_per_rt_s) * DT_S,
                    // Heading noise
                    sq(config.heading_process_noise_rad_per_rt_meter) *
                            old_robot_from_new_robot.arclength() +
                        sq(config.heading_process_noise_rad_per_rt_s) * DT_S,
                };

                out.local_from_robot = out.local_from_robot * old_robot_from_new_robot;
                const Eigen::Matrix3d dynamics_jac_wrt_state =
                    old_robot_from_new_robot.inverse().Adj();
                out.cov_in_robot =
                    dynamics_jac_wrt_state * out.cov_in_robot * dynamics_jac_wrt_state.transpose() +
                    process_noise_in_robot;
            }

            // Measurement Update
            {
                std::cout << "Robot Pos: " << out.local_from_robot.translation().transpose()
                          << std::endl;
                for (const auto beacon_id : est.beacon_ids) {
                    const Eigen::Vector2d beacon_in_local = est.beacon_in_local(beacon_id).value();
                    const Eigen::Vector2d beacon_in_robot =
                        out.local_from_robot.inverse() * beacon_in_local;
                    if (beacon_in_robot.norm() > max_sensor_range_m) {
                        continue;
                    }
                    std::cout << "In range of beacon " << beacon_id << std::endl;
                }
            }
        }
        return out;
    };
}
}  // namespace

double distance_to(const Eigen::Vector2d &pt_in_local, const RobotBelief &belief) {
    const Eigen::Vector2d pt_in_robot = belief.local_from_robot.inverse() * pt_in_local;
    return pt_in_robot.norm();
}

double uncertainty_size(const RobotBelief &belief) {
    // Should this be the covariance about the map frame
    return belief.cov_in_robot.determinant();
}

bool operator==(const RobotBelief &a, const RobotBelief &b) {
    constexpr double TOL = 1e-6;
    const auto mean_diff = (a.local_from_robot.log() - b.local_from_robot.log()).norm();
    const auto cov_diff = (a.cov_in_robot - b.cov_in_robot).norm();

    const bool is_mean_near = mean_diff < TOL;
    const bool is_cov_near = cov_diff < (TOL * TOL);
    return is_mean_near && is_cov_near;
}

std::optional<planning::BRMPlan<RobotBelief>> compute_belief_road_map_plan(
    const planning::RoadMap &road_map, const EkfSlam &ekf, const Eigen::Vector2d &goal_state,
    const double max_sensor_range_m) {
    const auto &estimate = ekf.estimate();

    const RobotBelief initial_belief = {
        .local_from_robot = estimate.local_from_robot(),
        .cov_in_robot = estimate.robot_cov(),
    };
    const auto belief_updater = make_belief_updater(road_map, goal_state, max_sensor_range_m, ekf);
    return planning::plan<RobotBelief>(road_map, initial_belief, belief_updater, goal_state);
}
}  // namespace robot::experimental::beacon_sim

namespace std {
template <>
struct std::hash<robot::experimental::beacon_sim::RobotBelief> {
    std::size_t operator()(const robot::experimental::beacon_sim::RobotBelief &belief) const {
        std::hash<double> double_hasher;
        // This is probably a terrible hash function
        return (double_hasher(belief.local_from_robot.log().norm()) << 3) ^
               double_hasher(belief.cov_in_robot.determinant());
    }
};
}  // namespace std
