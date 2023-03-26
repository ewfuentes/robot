
#include "experimental/beacon_sim/belief_road_map_planner.hh"

#include <random>
#include <unordered_map>

#include "Eigen/Core"
#include "common/liegroups/se2.hh"
#include "common/math/redheffer_star.hh"
#include "experimental/beacon_sim/correlated_beacons.hh"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/generate_observations.hh"
#include "experimental/beacon_sim/robot.hh"

namespace robot::experimental::beacon_sim {
namespace {
using ScatteringTransformMatrix =
    Eigen::Matrix<double, 2 * liegroups::SE2::DoF, 2 * liegroups::SE2::DoF>;

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

Eigen::Matrix3d compute_process_noise(const EkfSlamConfig &config, const double dt_s,
                                      const double arclength_m) {
    const auto sq = [](const double x) { return x * x; };

    return Eigen::DiagonalMatrix<double, 3>(
        sq(config.along_track_process_noise_m_per_rt_meter) * arclength_m +
            sq(config.pos_process_noise_m_per_rt_s) * dt_s,
        sq(config.cross_track_process_noise_m_per_rt_meter) * arclength_m +
            sq(config.pos_process_noise_m_per_rt_s) * dt_s,
        sq(config.heading_process_noise_rad_per_rt_meter) * arclength_m +
            sq(config.heading_process_noise_rad_per_rt_s) * dt_s);
}

std::vector<BeaconObservation> generate_observations(const liegroups::SE2 &local_from_robot,
                                                     const EkfSlamEstimate &estimate,
                                                     const double max_sensor_range_m) {
    std::vector<BeaconObservation> out;
    const RobotState robot_state(local_from_robot);
    const ObservationConfig config = {
        .range_noise_std_m = std::nullopt,
        .max_sensor_range_m = max_sensor_range_m,
    };

    for (const int beacon_id : estimate.beacon_ids) {
        std::mt19937 gen(0);
        Beacon beacon = {
            .id = beacon_id,
            .pos_in_local = estimate.beacon_in_local(beacon_id).value(),
        };
        const auto &maybe_observation =
            generate_observation(beacon, robot_state, config, make_in_out(gen));
        if (maybe_observation.has_value()) {
            out.push_back(maybe_observation.value());
        }
    }
    return out;
}

Eigen::MatrixXd build_measurement_noise(const int num_observations,
                                        const double range_measurement_noise_m,
                                        const double bearing_measurement_noise_rad) {
    Eigen::MatrixXd noise = Eigen::MatrixXd::Identity(2 * num_observations, 2 * num_observations);

    noise(Eigen::seq(0, Eigen::last, 2), Eigen::all) *=
        range_measurement_noise_m * range_measurement_noise_m;
    noise(Eigen::seq(1, Eigen::last, 2), Eigen::all) *=
        bearing_measurement_noise_rad * bearing_measurement_noise_rad;

    return noise;
}

ScatteringTransformMatrix compute_process_transform(
    const Eigen::Matrix3d &process_noise, const liegroups::SE2 &old_robot_from_new_robot) {
    const Eigen::Matrix3d dynamics_jac_wrt_state = old_robot_from_new_robot.inverse().Adj();

    // See equation 69 from
    // https://dspace.mit.edu/bitstream/handle/1721.1/58752/Roy_The%20belief.pdf

    return (ScatteringTransformMatrix() << dynamics_jac_wrt_state, process_noise,
            Eigen::Matrix3d::Zero(), dynamics_jac_wrt_state.transpose())
        .finished();
}

ScatteringTransformMatrix compute_measurement_transform(const liegroups::SE2 &local_from_robot,
                                                        const EkfSlamConfig &ekf_config,
                                                        const EkfSlamEstimate &ekf_estimate,
                                                        const BeaconPotential &beacon_potential,
                                                        const double max_sensor_range_m) {
    // Simulate observations
    const auto observations =
        generate_observations(local_from_robot, ekf_estimate, max_sensor_range_m);

    if (observations.empty()) {
        return ScatteringTransformMatrix::Identity();
    }

    // See equation 70 from
    // https://dspace.mit.edu/bitstream/handle/1721.1/58752/Roy_The%20belief.pdf

    // Generate the observation matrix
    const detail::UpdateInputs inputs =
        detail::compute_measurement_and_prediction(observations, ekf_estimate, local_from_robot);

    const Eigen::MatrixXd observation_matrix =
        inputs.observation_matrix(Eigen::all, Eigen::seqN(0, liegroups::SE2::DoF));

    // Generate the measurement noise
    const Eigen::MatrixXd measurement_noise = build_measurement_noise(
        inputs.observation_matrix.rows() / 2, ekf_config.range_measurement_noise_m,
        ekf_config.bearing_measurement_noise_rad);

    (void)beacon_potential;

    return (ScatteringTransformMatrix() << Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Zero(),
            -observation_matrix.transpose() * measurement_noise.inverse() * observation_matrix,
            Eigen::Matrix3d::Identity())
        .finished();
}

planning::BeliefUpdater<RobotBelief> make_belief_updater(const planning::RoadMap &road_map,
                                                         const Eigen::Vector2d &goal_state,
                                                         const double max_sensor_range_m,
                                                         const EkfSlam &ekf,
                                                         const BeaconPotential &beacon_potential) {
    std::unordered_map<DirectedEdge, detail::ScatteringTransform, DirectedEdgeHash>
        edge_transform_cache;
    return [&road_map, goal_state, max_sensor_range_m, &ekf,
            edge_transform_cache = std::move(edge_transform_cache),
            &beacon_potential](const RobotBelief &initial_belief, const int start_idx,
                               const int end_idx) mutable -> RobotBelief {
        // Get the belief edge transform, optionally updating the cache
        const DirectedEdge edge = {
            .source = start_idx,
            .destination = end_idx,
            .initial_heading_in_local = initial_belief.local_from_robot.so2().log(),
        };
        const auto cache_iter = edge_transform_cache.find(edge);
        const bool is_in_cache = cache_iter != edge_transform_cache.end();
        const auto end_pos_in_local = end_idx < 0 ? goal_state : road_map.points.at(end_idx);
        const detail::ScatteringTransform &transform =
            is_in_cache ? cache_iter->second
                        : detail::compute_edge_belief_transform(
                              initial_belief.local_from_robot, end_pos_in_local, ekf.config(),
                              ekf.estimate(), beacon_potential, max_sensor_range_m);
        if (!is_in_cache) {
            // Add the transform to the cache in case it's missing
            edge_transform_cache[edge] = transform;
        }

        // Compute the new covariance
        // [- new_cov] = [I cov] * edge_transform
        // [-       -]   [0   I]
        // new_cov  = A * B^-1
        const int cov_dim = initial_belief.cov_in_robot.rows();
        Eigen::MatrixXd input = Eigen::MatrixXd::Identity(2 * cov_dim, 2 * cov_dim);
        input.topRightCorner(cov_dim, cov_dim) = initial_belief.cov_in_robot;

        const ScatteringTransformMatrix result =
            math::redheffer_star(input, transform.cov_transform);

        const Eigen::MatrixXd new_cov_in_robot = result.topRightCorner(cov_dim, cov_dim);

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
    const planning::RoadMap &road_map, const EkfSlam &ekf, const BeaconPotential &beacon_potential,
    const Eigen::Vector2d &goal_state, const double max_sensor_range_m,
    const int num_start_connections, const int num_goal_connections,
    const double uncertainty_tolerance) {
    const auto &estimate = ekf.estimate();

    const RobotBelief initial_belief = {
        .local_from_robot = estimate.local_from_robot(),
        .cov_in_robot = estimate.robot_cov(),
    };
    const auto belief_updater =
        make_belief_updater(road_map, goal_state, max_sensor_range_m, ekf, beacon_potential);
    return planning::plan<RobotBelief>(road_map, initial_belief, belief_updater, goal_state,
                                       num_start_connections, num_goal_connections,
                                       uncertainty_tolerance);
}

namespace detail {
ScatteringTransform compute_edge_belief_transform(const liegroups::SE2 &local_from_robot,
                                                  const Eigen::Vector2d &end_state_in_local,
                                                  const EkfSlamConfig &ekf_config,
                                                  const EkfSlamEstimate &ekf_estimate,
                                                  const BeaconPotential &beacon_potential,
                                                  const double max_sensor_range_m) {
    constexpr double DT_S = 0.5;
    constexpr double VELOCITY_MPS = 2.0;
    constexpr double ANGULAR_VELOCITY_RADPS = 2.0;

    liegroups::SE2 local_from_new_robot = local_from_robot;
    constexpr double TOL = 1e-6;
    ScatteringTransformMatrix scattering_transform = ScatteringTransformMatrix::Identity();
    for (Eigen::Vector2d end_in_robot = local_from_robot.inverse() * end_state_in_local;
         end_in_robot.norm() > TOL;
         end_in_robot = local_from_new_robot.inverse() * end_state_in_local) {
        // Move towards the goal
        const liegroups::SE2 old_robot_from_new_robot = [&]() {
            const double angle_to_goal_rad = std::atan2(end_in_robot.y(), end_in_robot.x());
            if (std::abs(angle_to_goal_rad) > TOL) {
                // First turn to face the goal
                constexpr double MAX_ANGLE_STEP_RAD = DT_S * ANGULAR_VELOCITY_RADPS;
                const double angle_step_rad = std::copysign(
                    std::min(std::abs(angle_to_goal_rad), MAX_ANGLE_STEP_RAD), angle_to_goal_rad);
                return liegroups::SE2::rot(angle_step_rad);
            } else {
                // Then drive towards the goal
                constexpr double MAX_DIST_STEP_M = DT_S * VELOCITY_MPS;
                const double dist_step_m = std::min(end_in_robot.x(), MAX_DIST_STEP_M);
                return liegroups::SE2::transX(dist_step_m);
            }
        }();

        // Update the mean
        local_from_new_robot = local_from_new_robot * old_robot_from_new_robot;

        // Compute the process update for the covariance
        const Eigen::Matrix3d process_noise_in_robot =
            compute_process_noise(ekf_config, DT_S, old_robot_from_new_robot.arclength());

        const ScatteringTransformMatrix process_transform =
            compute_process_transform(process_noise_in_robot, old_robot_from_new_robot);

        // Compute the measurement update for the covariance
        const ScatteringTransformMatrix measurement_transform = compute_measurement_transform(
            local_from_new_robot, ekf_config, ekf_estimate, beacon_potential, max_sensor_range_m);

        scattering_transform = math::redheffer_star(measurement_transform, scattering_transform);
        scattering_transform = math::redheffer_star(process_transform, scattering_transform);
    }

    return ScatteringTransform{
        // Avoid numerical issues by directly setting the translation
        .local_from_robot = liegroups::SE2(local_from_new_robot.so2().log(), end_state_in_local),
        .cov_transform = scattering_transform,
    };
}

}  // namespace detail
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
