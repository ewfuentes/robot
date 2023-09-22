
#include "experimental/beacon_sim/make_belief_updater.hh"

#include "common/math/redheffer_star.hh"
#include "common/geometry/nearest_point_on_segment.hh"

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

std::vector<int> find_beacons_along_path(const Eigen::Vector2d &start_in_local,
                                         const Eigen::Vector2d &end_in_local,
                                         const EkfSlamEstimate &est,
                                         const double max_sensor_range_m) {
    std::unordered_set<int> beacon_ids;
    for (const int beacon_id : est.beacon_ids) {
        const Eigen::Vector2d beacon_in_local = est.beacon_in_local(beacon_id).value();

        const auto result = geometry::nearest_point_on_segment_result(start_in_local, end_in_local,
                                                                      beacon_in_local);

        const double dist_m = (beacon_in_local - result.nearest_pt_in_frame).norm();
        if (dist_m < max_sensor_range_m) {
            beacon_ids.insert(beacon_id);
        }
    }
    std::vector<int> out;
    std::copy(beacon_ids.begin(), beacon_ids.end(), std::back_inserter(out));
    std::sort(out.begin(), out.end());
    return out;
}


std::vector<LogMarginal> sample_log_marginals(const std::vector<LogMarginal> all_marginals,
                                              const int num_samples) {
    std::mt19937 gen(0);
    std::uniform_real_distribution<> dist;
    std::vector<double> rand_nums{};
    for (int i = 0; i < num_samples; i++) {
        rand_nums.push_back(dist(gen));
    }
    std::sort(rand_nums.begin(), rand_nums.end());

    double accumulated_prob = 0.0;
    std::vector<LogMarginal> out;
    auto rand_num_start = rand_nums.begin();

    for (const auto &marginal : all_marginals) {
        accumulated_prob += std::exp(marginal.log_marginal);

        const auto new_start = std::find_if(
            rand_num_start, rand_nums.end(),
            [&accumulated_prob](const double rand_num) { return rand_num > accumulated_prob; });

        const int counts = std::distance(rand_num_start, new_start);
        if (counts > 0) {
            out.push_back({
                .present_beacons = marginal.present_beacons,
                .log_marginal = std::log(counts),
            });
        }
        rand_num_start = new_start;
        if (rand_num_start == rand_nums.end()) {
            break;
        }
    }
    return out;
}


}


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

std::vector<BeaconObservation> generate_observations(
    const liegroups::SE2 &local_from_robot, const EkfSlamEstimate &estimate,
    const std::optional<std::vector<int>> &available_beacons, const double max_sensor_range_m) {
    std::vector<BeaconObservation> out;
    const RobotState robot_state(local_from_robot);
    const ObservationConfig config = {
        .range_noise_std_m = std::nullopt,
        .max_sensor_range_m = max_sensor_range_m,
    };

    const std::vector<int> &beacon_list =
        available_beacons.has_value() ? available_beacons.value() : estimate.beacon_ids;

    for (const int beacon_id : beacon_list) {
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

EdgeTransform::Matrix compute_process_transform(
    const Eigen::Matrix3d &process_noise, const liegroups::SE2 &old_robot_from_new_robot) {
    const Eigen::Matrix3d dynamics_jac_wrt_state = old_robot_from_new_robot.inverse().Adj();

    // See equation 69 from
    // https://dspace.mit.edu/bitstream/handle/1721.1/58752/Roy_The%20belief.pdf

    return (EdgeTransform::Matrix() << dynamics_jac_wrt_state, process_noise,
            Eigen::Matrix3d::Zero(), dynamics_jac_wrt_state.transpose())
        .finished();
}

EdgeTransform::Matrix compute_measurement_transform(
    const liegroups::SE2 &local_from_robot, const EkfSlamConfig &ekf_config,
    const EkfSlamEstimate &ekf_estimate, const std::optional<std::vector<int>> &available_beacons,
    const double max_sensor_range_m) {
    // Simulate observations
    const auto observations = generate_observations(local_from_robot, ekf_estimate,
                                                    available_beacons, max_sensor_range_m);

    if (observations.empty()) {
        return EdgeTransform::Matrix::Identity();
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

    return (EdgeTransform::Matrix() << Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Zero(),
            -observation_matrix.transpose() * measurement_noise.inverse() * observation_matrix,
            Eigen::Matrix3d::Identity())
        .finished();
}


EdgeTransform compute_edge_belief_transform(const liegroups::SE2 &local_from_robot,
                                            const Eigen::Vector2d &end_state_in_local,
                                            const EkfSlamConfig &ekf_config,
                                            const EkfSlamEstimate &ekf_estimate,
                                            const BeaconPotential &beacon_potential,
                                            const double max_sensor_range_m,
                                            const int max_num_transforms) {
    // Find all beacons along the path
    const std::vector<int> nearby_beacon_ids = find_beacons_along_path(
        local_from_robot.translation(), end_state_in_local, ekf_estimate, max_sensor_range_m);

    // Find the beacons that are part of the potential
    std::unordered_set<int> potential_beacons(beacon_potential.members().begin(),
                                              beacon_potential.members().end());

    std::vector<int> nearby_potential_beacons;
    std::vector<int> nearby_forever_beacons;
    for (const int beacon_id : nearby_beacon_ids) {
        if (potential_beacons.contains(beacon_id)) {
            nearby_potential_beacons.push_back(beacon_id);
        } else {
            nearby_forever_beacons.push_back(beacon_id);
        }
    }

    const auto all_log_marginals = beacon_potential.compute_log_marginals(nearby_potential_beacons);
    const auto log_marginals = static_cast<int>(all_log_marginals.size()) > max_num_transforms
                                   ? sample_log_marginals(all_log_marginals, max_num_transforms)
                                   : all_log_marginals;

    std::vector<double> weights;
    std::vector<EdgeTransform::Matrix> transforms;
    liegroups::SE2 local_from_end_robot;
    for (const auto &log_marginal : log_marginals) {
        std::vector<int> present_beacons;
        std::copy(nearby_forever_beacons.begin(), nearby_forever_beacons.end(),
                  std::back_inserter(present_beacons));
        std::copy(log_marginal.present_beacons.begin(), log_marginal.present_beacons.end(),
                  std::back_inserter(present_beacons));
        const auto &[local_from_final_robot, scattering_transform] =
            compute_edge_belief_transform(local_from_robot, end_state_in_local, ekf_config,
                                          ekf_estimate, present_beacons, max_sensor_range_m);

        weights.push_back(std::exp(log_marginal.log_marginal));
        transforms.push_back(scattering_transform);
        local_from_end_robot = local_from_final_robot;
    }

    return EdgeTransform{
        .local_from_robot = local_from_end_robot,
        .weight = std::move(weights),
        .transforms = std::move(transforms),
    };
}

planning::BeliefUpdater<RobotBelief> make_belief_updater(const planning::RoadMap &road_map,
                                                         const Eigen::Vector2d &goal_state,
                                                         const double max_sensor_range_m,
                                                         const int max_num_transforms,
                                                         const EkfSlam &ekf,
                                                         const BeaconPotential &beacon_potential) {
    std::unordered_map<DirectedEdge, EdgeTransform, DirectedEdgeHash> edge_transform_cache;
    return [&road_map, goal_state, max_sensor_range_m, &ekf,
            edge_transform_cache = std::move(edge_transform_cache), &beacon_potential,
            max_num_transforms](const RobotBelief &initial_belief, const int start_idx,
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
        const EdgeTransform &edge_transform =
            is_in_cache
                ? cache_iter->second
                : compute_edge_belief_transform(
                      initial_belief.local_from_robot, end_pos_in_local, ekf.config(),
                      ekf.estimate(), beacon_potential, max_sensor_range_m, max_num_transforms);
        if (!is_in_cache) {
            // Add the transform to the cache in case it's missing
            edge_transform_cache[edge] = edge_transform;
        }

        // Compute the new covariance
        // [- new_cov] = [I cov] * edge_transform
        // [-       -]   [0   I]
        // new_cov  = A * B^-1
        const int cov_dim = initial_belief.cov_in_robot.rows();
        Eigen::MatrixXd input = Eigen::MatrixXd::Identity(2 * cov_dim, 2 * cov_dim);
        input.topRightCorner(cov_dim, cov_dim) = initial_belief.cov_in_robot;

        Eigen::MatrixXd new_cov_in_robot = Eigen::MatrixXd::Zero(cov_dim, cov_dim);
        const double weight_total =
            std::accumulate(edge_transform.weight.begin(), edge_transform.weight.end(), 0.0);
        for (int i = 0; i < static_cast<int>(edge_transform.weight.size()); i++) {
            const double weight = edge_transform.weight.at(i) / weight_total;
            const auto &cov_transform = edge_transform.transforms.at(i);
            const EdgeTransform::Matrix result = math::redheffer_star(input, cov_transform);
            const Eigen::MatrixXd sample_cov_in_robot = result.topRightCorner(cov_dim, cov_dim);
            new_cov_in_robot += weight * sample_cov_in_robot;
        }

        return RobotBelief{
            .local_from_robot = edge_transform.local_from_robot,
            .cov_in_robot = new_cov_in_robot,
        };
    };
}

planning::BeliefUpdater<RobotBelief> make_belief_updater(const planning::RoadMap &road_map,
                                                         const Eigen::Vector2d &goal_state,
                                                         const double max_sensor_range_m,
                                                         const EkfSlam &ekf,
                                                         const std::vector<int> &present_beacons) {
    std::unordered_map<DirectedEdge, std::tuple<liegroups::SE2, EdgeTransform::Matrix>,
                       DirectedEdgeHash>
        edge_transform_cache;

    return [&road_map, goal_state, max_sensor_range_m, &ekf,
            edge_transform_cache = std::move(edge_transform_cache),
            &present_beacons](const RobotBelief &initial_belief, const int start_idx,
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
        const auto &[local_from_new_robot, edge_transform] =
            is_in_cache ? cache_iter->second
                        : compute_edge_belief_transform(
                              initial_belief.local_from_robot, end_pos_in_local, ekf.config(),
                              ekf.estimate(), present_beacons, max_sensor_range_m);
        if (!is_in_cache) {
            // Add the transform to the cache in case it's missing
            edge_transform_cache[edge] = std::make_tuple(local_from_new_robot, edge_transform);
        }

        // Compute the new covariance
        // [- new_cov] = [I cov] * edge_transform
        // [-       -]   [0   I]
        // new_cov  = A * B^-1
        const int cov_dim = initial_belief.cov_in_robot.rows();
        Eigen::MatrixXd input = Eigen::MatrixXd::Identity(2 * cov_dim, 2 * cov_dim);
        input.topRightCorner(cov_dim, cov_dim) = initial_belief.cov_in_robot;

        const EdgeTransform::Matrix result = math::redheffer_star(input, edge_transform);
        const Eigen::MatrixXd new_cov_in_robot = result.topRightCorner(cov_dim, cov_dim);

        return RobotBelief{
            .local_from_robot = local_from_new_robot,
            .cov_in_robot = new_cov_in_robot,
        };
    };
}

}
