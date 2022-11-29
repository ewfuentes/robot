
#include "experimental/beacon_sim/belief_road_map_planner.hh"

#include <iostream>
#include <random>
#include <unordered_map>

#include "Eigen/Core"
#include "experimental/beacon_sim/generate_observations.hh"
#include "experimental/beacon_sim/robot.hh"

namespace robot::experimental::beacon_sim {
namespace {
struct DirectedEdge {
    int source;
    int destination;
    double initial_heading_in_local;

    bool operator==(const DirectedEdge &other) const {
        return source == other.source && destination == other.destination &&
               initial_heading_in_local == other.initial_heading_in_local;
    }
};

struct DirectedEdgeHash {
    size_t operator()(const DirectedEdge &edge) const {
        std::hash<int> int_hasher;
        std::hash<double> double_hasher;
        return (int_hasher(edge.source) << 3) ^ int_hasher(edge.destination) ^
               (double_hasher(edge.initial_heading_in_local) << 5);
    }
};

struct EdgeBeliefTransform {
    liegroups::SE2 local_from_robot;
    Eigen::Matrix2d cov_transform;
};

// std::vector<BeaconObservation> generate_observations(const liegroups::SE2 &local_from_robot,
//                                                      const EkfSlamEstimate &estimate,
//                                                      const double max_sensor_range_m) {
//     std::vector<BeaconObservation> out;
//     const RobotState robot_state(local_from_robot);
//     const ObservationConfig config = {
//         .range_noise_std_m = std::nullopt,
//         .max_sensor_range_m = max_sensor_range_m,
//     };
//
//     for (const int beacon_id : estimate.beacon_ids) {
//         std::mt19937 gen(0);
//         Beacon beacon = {
//             .id = beacon_id,
//             .pos_in_local = estimate.beacon_in_local(beacon_id).value(),
//         };
//         const auto &maybe_observation =
//             generate_observation(beacon, robot_state, config, make_in_out(gen));
//         if (maybe_observation.has_value()) {
//             out.push_back(maybe_observation.value());
//         }
//     }
//     return out;
// }
//
// Eigen::MatrixXd build_measurement_noise(const int num_observations,
//                                         const double range_measurement_noise_m,
//                                         const double bearing_measurement_noise_rad) {
//     Eigen::MatrixXd noise = Eigen::MatrixXd::Identity(2 * num_observations, 2 *
//     num_observations);
//
//     noise(Eigen::seq(0, Eigen::last, 2), Eigen::all) *=
//         range_measurement_noise_m * range_measurement_noise_m;
//     noise(Eigen::seq(1, Eigen::last, 2), Eigen::all) *=
//         bearing_measurement_noise_rad * bearing_measurement_noise_rad;
//
//     return noise;
// }

EdgeBeliefTransform compute_edge_transform(const planning::RoadMap &road_map,
                                           const liegroups::SE2 &local_from_initial_robot,
                                           const Eigen::Vector2d &neighbor_pos_in_local) {
         constexpr double DT_S = 1.0;
         constexpr double VELOCITY_MPS = 2.0;
         constexpr double ANGULAR_VELOCITY_RADPS = 2.0;
    //     RobotBelief out = initial_belief;
    //     const EkfSlamConfig &config = ekf.config();
    //     const EkfSlamEstimate &est = ekf.estimate();
    //
    //     while ((out.local_from_robot.inverse() * end_position).norm() > 1e-6) {
    //         // Prediction Update
    //         {
    //             // Compute the control action to get us closer to the goal
    //             const liegroups::SE2 old_robot_from_new_robot = [&]() {
    //                 constexpr double ANGLE_TOL = 1e-6;
    //                 const Eigen::Vector2d goal_in_robot = out.local_from_robot.inverse() *
    //                 end_position; const double angle_to_goal_rad = std::atan2(goal_in_robot.y(),
    //                 goal_in_robot.x()); if (std::abs(angle_to_goal_rad) > ANGLE_TOL) {
    //                     constexpr double MAX_STEP_TURN_RAD = ANGULAR_VELOCITY_RADPS * DT_S;
    //                     const double step_turn_amount_rad =
    //                         std::copysign(std::min(std::abs(angle_to_goal_rad),
    //                         MAX_STEP_TURN_RAD),
    //                                       angle_to_goal_rad);
    //                     return liegroups::SE2::rot(step_turn_amount_rad);
    //                 } else {
    //                     constexpr double MAX_STEP_DIST_M = VELOCITY_MPS * DT_S;
    //                     const double dist_to_goal_m = goal_in_robot.norm();
    //                     const double step_travel_m = std::min(dist_to_goal_m, MAX_STEP_DIST_M);
    //                     return liegroups::SE2::transX(step_travel_m);
    //                 }
    //             }();
    //
    //             const auto sq = [](const double x) { return x * x; };
    //             const Eigen::Matrix3d process_noise_in_robot = Eigen::DiagonalMatrix<double, 3>{
    //                 // Along track noise
    //                 sq(config.along_track_process_noise_m_per_rt_meter) *
    //                         old_robot_from_new_robot.arclength() +
    //                     sq(config.pos_process_noise_m_per_rt_s) * DT_S,
    //                 // Cross track noise
    //                 sq(config.cross_track_process_noise_m_per_rt_meter) *
    //                         old_robot_from_new_robot.arclength() +
    //                     sq(config.pos_process_noise_m_per_rt_s) * DT_S,
    //                 // Heading noise
    //                 sq(config.heading_process_noise_rad_per_rt_meter) *
    //                         old_robot_from_new_robot.arclength() +
    //                     sq(config.heading_process_noise_rad_per_rt_s) * DT_S,
    //             };
    //
    //             out.local_from_robot = out.local_from_robot * old_robot_from_new_robot;
    //             const Eigen::Matrix3d dynamics_jac_wrt_state =
    //             old_robot_from_new_robot.inverse().Adj(); out.cov_in_robot =
    //                 dynamics_jac_wrt_state * out.cov_in_robot *
    //                 dynamics_jac_wrt_state.transpose() + process_noise_in_robot;
    //         }
    //
    //         // Measurement Update
    //         {
    //             const std::vector<BeaconObservation> observations =
    //                 generate_observations(out.local_from_robot, ekf.estimate(),
    //                 max_sensor_range_m);
    //
    //             if (observations.size() == 0) {
    //                 continue;
    //             }
    //
    //             EkfSlamEstimate updated_est = est;
    //
    //             updated_est.mean.head<2>() = out.local_from_robot.translation();
    //             updated_est.mean(2) = out.local_from_robot.so2().log();
    //
    //             const detail::UpdateInputs inputs =
    //                 detail::compute_measurement_and_prediction(observations, updated_est);
    //
    //             const Eigen::MatrixXd observation_matrix =
    //                 inputs.observation_matrix(Eigen::all, Eigen::seqN(0, 3));
    //
    //             const Eigen::MatrixXd measurement_noise =
    //                 build_measurement_noise(observations.size(),
    //                 config.range_measurement_noise_m,
    //                                         config.bearing_measurement_noise_rad);
    //
    //             const Eigen::MatrixXd innovation_cov =
    //                 observation_matrix * out.cov_in_robot * observation_matrix.transpose() +
    //                 measurement_noise;
    //             const Eigen::MatrixXd kalman_gain =
    //                 out.cov_in_robot * observation_matrix.transpose() * innovation_cov.inverse();
    //             const Eigen::Matrix3d I_min_KH =
    //                 Eigen::Matrix3d::Identity() - kalman_gain * observation_matrix;
    //             out.cov_in_robot = I_min_KH * out.cov_in_robot;
    //         }
    //     }
    return EdgeBeliefTransform{
        .local_from_robot = liegroups::SE2(),
        .cov_transform = Eigen::MatrixXd(),
    };
}

planning::BeliefUpdater<RobotBelief> make_belief_updater(const planning::RoadMap &road_map,
                                                         const Eigen::Vector2d &goal_state,
                                                         const double max_sensor_range_m,
                                                         const EkfSlam &ekf) {
    std::unordered_map<DirectedEdge, EdgeBeliefTransform, DirectedEdgeHash> edge_transform_cache;
    return [&road_map, goal_state, max_sensor_range_m, &ekf,
            edge_transform_cache = std::move(edge_transform_cache)](
               const RobotBelief &initial_belief, const int start_idx, const int end_idx) mutable {
        (void)road_map;
        (void)goal_state;
        (void)max_sensor_range_m;
        (void)ekf;
        // Get the belief edge transform, optionally updating the cache
        const DirectedEdge edge = {
            .source = start_idx,
            .destination = end_idx,
            .initial_heading_in_local = initial_belief.local_from_robot.so().log(),
        };
        const auto cache_iter = edge_transform_cache.find(edge);
        const bool is_in_cache = cache_iter != edge_transform_cache.end();
        const auto end_pos_in_local = end_idx < 0 ? goal_state : road_map.points.at(end_idx);
        const EdgeBeliefTransform &transform =
            is_in_cache ? cache_iter->second
                        : compute_edge_transform(road_map, initial_belief.local_from_robot,
                                                 end_pos_in_local);
        if (!is_in_cache) {
            // Add the transform to the cache in case it's missing
            edge_transform_cache[edge] = transform;
        }

        // Compute the new covariance
        // [A] = [C D][cov]
        // [B]   [E F][ I ]
        // new_cov  = A * B^-1
        const int cov_dim = initial_belief.cov_in_robot.rows();
        Eigen::MatrixXd input = Eigen::MatrixXd::Zero(2 * cov_dim, cov_dim);
        input.topLeftCorner(cov_dim, cov_dim) = initial_belief.cov_in_robot;
        input.bottomLeftCorner(cov_dim, cov_dim) = Eigen::MatrixXd::Identity(cov_dim, cov_dim);

        const Eigen::MatrixXd transformed = transform.cov_transform * input;
        const Eigen::MatrixXd numerator = transformed.topLeftCorner(cov_dim, cov_dim);
        const Eigen::MatrixXd denominator = transformed.bottomLeftCorner(cov_dim, cov_dim);

        const Eigen::MatrixXd new_cov_in_robot = numerator * denominator.inverse();

        return RobotBelief{
            .local_from_robot = transform.local_from_robot,
            .cov_in_robot = new_cov_in_robot,
        };
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
    constexpr double TOL = 1e-3;
    // Note that we don't consider covariance
    const auto mean_diff =
        (a.local_from_robot.translation() - b.local_from_robot.translation()).norm();

    const bool is_mean_near = mean_diff < TOL;
    return is_mean_near;
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
struct hash<robot::experimental::beacon_sim::RobotBelief> {
    std::size_t operator()(const robot::experimental::beacon_sim::RobotBelief &belief) const {
        std::hash<double> double_hasher;
        // This is probably a terrible hash function
        // Note that we don't consider the heading or covariance
        return double_hasher(belief.local_from_robot.translation().norm());
    }
};
}  // namespace std
