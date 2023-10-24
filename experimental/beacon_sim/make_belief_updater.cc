
#include "experimental/beacon_sim/make_belief_updater.hh"

#include <bits/chrono_io.h>

#include <algorithm>
#include <iostream>
#include <iterator>

#include "common/check.hh"
#include "common/geometry/nearest_point_on_segment.hh"
#include "common/math/combinations.hh"
#include "common/math/redheffer_star.hh"
#include "experimental/beacon_sim/robot_belief.hh"

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

struct BeaconGroups {
    std::vector<int> beacons_in_potential;
    std::vector<int> beacons_always_present;
};

BeaconGroups split_beacons_into_groups(std::vector<int> beacon_list,
                                       std::vector<int> all_beacons_in_potential) {
    std::unordered_set<int> all_beacons_in_potential_set(all_beacons_in_potential.begin(),
                                                         all_beacons_in_potential.end());

    std::vector<int> beacons_in_potential;
    std::vector<int> beacons_always_present;
    for (const int beacon_id : beacon_list) {
        if (all_beacons_in_potential_set.contains(beacon_id)) {
            beacons_in_potential.push_back(beacon_id);
        } else {
            beacons_always_present.push_back(beacon_id);
        }
    }

    return {
        .beacons_in_potential = std::move(beacons_in_potential),
        .beacons_always_present = std::move(beacons_always_present),
    };
}

std::string configuration_to_key(const std::vector<std::tuple<int, bool>> &beacon_config,
                                 const std::vector<int> &all_beacons) {
    std::string out(all_beacons.size(), '?');
    for (const auto &[beacon_id, is_present] : beacon_config) {
        const auto iter = std::find(all_beacons.begin(), all_beacons.end(), beacon_id);
        const int idx = std::distance(all_beacons.begin(), iter);
        out[idx] = is_present ? '1' : '0';
    }
    return out;
}

std::string merge_configurations(const std::string &a, const std::string &b) {
    std::string out = a;
    CHECK(a.size() == b.size());

    for (int i = 0; i < static_cast<int>(a.size()); i++) {
        const bool a_is_set = a.at(i) != '?';
        const bool b_is_set = b.at(i) != '?';

        CHECK(a_is_set ^ b_is_set || a.at(i) == b.at(i), "a and b are not mergable", i, a, b);
        if (b_is_set) {
            out.at(i) = b.at(i);
        }
    }
    return out;
}

template <typename T>
std::vector<typename T::const_iterator> find_consistent_configs(const std::string &config,
                                                                const T &transform_list) {
    // A config is consistent if every beacon that is mutually known matches
    std::vector<typename T::const_iterator> out;
    for (auto iter = transform_list.begin(); iter != transform_list.end(); iter++) {
        const auto &[transform_config, _] = *iter;
        bool is_consistent = true;
        for (int i = 0; i < static_cast<int>(config.size()); i++) {
            if (config.at(i) != transform_config.at(i)) {
                if (config.at(i) == '?' || transform_config.at('?')) {
                    continue;
                } else {
                    // Both config and the transform config are observed, but they don't match
                    // They are not consistent
                    is_consistent = false;
                    break;
                }
            }
        }
        if (is_consistent) {
            out.push_back(iter);
        }
    }
    return out;
}

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

}  // namespace

std::optional<TypedTransform> operator*(const TypedTransform &a, const TypedTransform &b) {
    if (a.index() != b.index()) {
        return std::nullopt;
    } else if (std::holds_alternative<ScatteringTransform<TransformType::INFORMATION>>(a)) {
        return std::make_optional(std::get<ScatteringTransform<TransformType::INFORMATION>>(a) *
                                  std::get<ScatteringTransform<TransformType::INFORMATION>>(b));
    } else {
        return std::make_optional(std::get<ScatteringTransform<TransformType::COVARIANCE>>(a) *
                                  std::get<ScatteringTransform<TransformType::COVARIANCE>>(b));
    }
}

Eigen::DiagonalMatrix<double, 3> compute_process_noise(const EkfSlamConfig &config,
                                                       const double dt_s,
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

TypedTransform compute_process_transform(const Eigen::Matrix3d &process_noise,
                                         const liegroups::SE2 &old_robot_from_new_robot,
                                         const TransformType type) {
    // See equation 69 from
    // https://dspace.mit.edu/bitstream/handle/1721.1/58752/Roy_The%20belief.pdf

    if (type == TransformType::COVARIANCE) {
        const Eigen::Matrix3d dynamics_jac_wrt_state = old_robot_from_new_robot.inverse().Adj();
        return ScatteringTransform<TransformType::COVARIANCE>(
            (ScatteringTransform<TransformType::COVARIANCE>() << dynamics_jac_wrt_state,
             process_noise, Eigen::Matrix3d::Zero(), dynamics_jac_wrt_state.transpose())
                .finished());
    } else {
        const Eigen::Matrix3d inv_dynamics_jac_wrt_state = old_robot_from_new_robot.Adj();
        return ScatteringTransform<TransformType::INFORMATION>(
            (ScatteringTransform<TransformType::INFORMATION>()
                 << inv_dynamics_jac_wrt_state.transpose(),
             Eigen::Matrix3d::Zero(),
             -inv_dynamics_jac_wrt_state * process_noise * inv_dynamics_jac_wrt_state.transpose(),
             inv_dynamics_jac_wrt_state)
                .finished());
    }
}

TypedTransform compute_measurement_transform(
    const liegroups::SE2 &local_from_robot, const EkfSlamConfig &ekf_config,
    const EkfSlamEstimate &ekf_estimate, const std::optional<std::vector<int>> &available_beacons,
    const double max_sensor_range_m, const TransformType type) {
    // Simulate observations
    const auto observations = generate_observations(local_from_robot, ekf_estimate,
                                                    available_beacons, max_sensor_range_m);

    if (observations.empty()) {
        if (type == TransformType::COVARIANCE) {
            return ScatteringTransform<TransformType::COVARIANCE>::Identity();
        } else {
            return ScatteringTransform<TransformType::INFORMATION>::Identity();
        }
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

    if (type == TransformType::COVARIANCE) {
        return ScatteringTransform<TransformType::COVARIANCE>(
            (ScatteringTransform<TransformType::COVARIANCE>() << Eigen::Matrix3d::Identity(),
             Eigen::Matrix3d::Zero(),
             -observation_matrix.transpose() * measurement_noise.inverse() * observation_matrix,
             Eigen::Matrix3d::Identity())
                .finished());
    } else {
        return ScatteringTransform<TransformType::INFORMATION>(
            (ScatteringTransform<TransformType::INFORMATION>() << Eigen::Matrix3d::Identity(),
             observation_matrix.transpose() * measurement_noise.inverse() * observation_matrix,
             Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Identity())
                .finished());
    }
}

EdgeTransform compute_edge_belief_transform(
    const liegroups::SE2 &local_from_robot, const Eigen::Vector2d &end_state_in_local,
    const EkfSlamConfig &ekf_config, const EkfSlamEstimate &ekf_estimate,
    const BeaconPotential &beacon_potential, const double max_sensor_range_m,
    const int max_num_transforms, const TransformType type) {
    // Find all beacons along the path
    const std::vector<int> nearby_beacon_ids = find_beacons_along_path(
        local_from_robot.translation(), end_state_in_local, ekf_estimate, max_sensor_range_m);

    // Find the beacons that are part of the potential
    const auto &[nearby_potential_beacons, nearby_forever_beacons] =
        split_beacons_into_groups(nearby_beacon_ids, beacon_potential.members());

    const auto all_log_marginals = beacon_potential.compute_log_marginals(nearby_potential_beacons);
    const auto log_marginals = static_cast<int>(all_log_marginals.size()) > max_num_transforms
                                   ? sample_log_marginals(all_log_marginals, max_num_transforms)
                                   : all_log_marginals;

    std::vector<double> weights;
    TypedTransformVector cov_transforms;
    liegroups::SE2 local_from_end_robot;
    for (const auto &log_marginal : log_marginals) {
        std::vector<int> present_beacons;
        std::copy(nearby_forever_beacons.begin(), nearby_forever_beacons.end(),
                  std::back_inserter(present_beacons));
        std::copy(log_marginal.present_beacons.begin(), log_marginal.present_beacons.end(),
                  std::back_inserter(present_beacons));
        const auto &[local_from_final_robot, scattering_transform] =
            compute_edge_belief_transform(local_from_robot, end_state_in_local, ekf_config,
                                          ekf_estimate, present_beacons, max_sensor_range_m, type);

        weights.push_back(std::exp(log_marginal.log_marginal));
        cov_transforms.push_back(scattering_transform);
        local_from_end_robot = local_from_final_robot;
    }

    return EdgeTransform{
        .local_from_robot = local_from_end_robot,
        .weight = std::move(weights),
        .transforms = std::move(cov_transforms),
    };
}

std::tuple<liegroups::SE2, std::vector<std::tuple<std::string, TypedTransform>>>
compute_edge_belief_transform(const liegroups::SE2 &local_from_robot,
                              const Eigen::Vector2d &end_state_in_local,
                              const EkfSlamConfig &ekf_config, const EkfSlamEstimate &ekf_estimate,
                              const BeaconPotential &beacon_potential,
                              const double max_sensor_range_m, const TransformType transform_type) {
    // Find all beacons that are within range of the straightline path of the start and end
    const std::vector<int> nearby_beacons = find_beacons_along_path(
        local_from_robot.translation(), end_state_in_local, ekf_estimate, max_sensor_range_m);

    // Find the beacons that are part of the potential
    const auto &[beacons_in_potential, always_present_beacons] =
        split_beacons_into_groups(nearby_beacons, beacon_potential.members());

    // Create a list of which beacons involved in order to create the key for each configuration
    std::vector<std::tuple<int, bool>> config_base;
    std::transform(beacons_in_potential.begin(), beacons_in_potential.end(),
                   std::back_inserter(config_base),
                   [](const int beacon_id) { return std::make_tuple(beacon_id, false); });
    std::vector<std::tuple<std::string, TypedTransform>> transform_list;
    liegroups::SE2 local_from_end_robot;
    // Iterate over combinations of the nearby beacons
    for (int num_beacons_present = 0;
         num_beacons_present <= static_cast<int>(beacons_in_potential.size());
         num_beacons_present++) {
        for (const auto &potential_beacon_idxs :
             math::combinations(beacons_in_potential.size(), num_beacons_present)) {
            std::vector<int> present_beacons(always_present_beacons.begin(),
                                             always_present_beacons.end());
            // create a copy of the base config
            auto current_config = config_base;
            for (const int beacon_idx : potential_beacon_idxs) {
                const int beacon_id = beacon_potential.members().at(beacon_idx);
                present_beacons.push_back(beacon_id);
                auto iter = std::find_if(current_config.begin(), current_config.end(),
                                         [beacon_id](const auto &config_element) {
                                             return std::get<int>(config_element) == beacon_id;
                                         });
                CHECK(
                    iter != current_config.end(),
                    "Unable to find present beacon idx in list of all nearby beacons in potential",
                    beacon_id, beacons_in_potential, beacon_potential.members());
                std::get<bool>(*iter) = true;
            }

            const auto &[config_local_from_end_robot, config_transform] =
                compute_edge_belief_transform(local_from_robot, end_state_in_local, ekf_config,
                                              ekf_estimate, present_beacons, max_sensor_range_m,
                                              transform_type);

            local_from_end_robot = config_local_from_end_robot;

            // set all beacons that are currently present to true
            transform_list.push_back(
                std::make_tuple(configuration_to_key(current_config, beacon_potential.members()),
                                config_transform));
        }
    }
    return std::make_tuple(local_from_end_robot, std::move(transform_list));
}

std::tuple<liegroups::SE2, TypedTransform> compute_edge_belief_transform(
    const liegroups::SE2 &local_from_robot, const Eigen::Vector2d &end_state_in_local,
    const EkfSlamConfig &ekf_config, const EkfSlamEstimate &ekf_estimate,
    const std::optional<std::vector<int>> &available_beacons, const double max_sensor_range_m,
    const TransformType type) {
    constexpr double DT_S = 0.5;
    constexpr double VELOCITY_MPS = 2.0;
    constexpr double ANGULAR_VELOCITY_RADPS = 2.0;

    liegroups::SE2 local_from_new_robot = local_from_robot;
    constexpr double TOL = 1e-6;
    TypedTransform scattering_transform;
    if (type == TransformType::COVARIANCE) {
        scattering_transform = ScatteringTransform<TransformType::COVARIANCE>::Identity();
    } else {
        scattering_transform = ScatteringTransform<TransformType::INFORMATION>::Identity();
    }

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

        const TypedTransform process_transform =
            compute_process_transform(process_noise_in_robot, old_robot_from_new_robot, type);

        // Compute the measurement update for the covariance
        const TypedTransform measurement_transform =
            compute_measurement_transform(local_from_new_robot, ekf_config, ekf_estimate,
                                          available_beacons, max_sensor_range_m, type);

        scattering_transform = (scattering_transform * process_transform).value();
        scattering_transform = (scattering_transform * measurement_transform).value();
    }

    return std::make_tuple(liegroups::SE2(local_from_new_robot.so2().log(), end_state_in_local),
                           scattering_transform);
}

planning::BeliefUpdater<RobotBelief> make_belief_updater(const planning::RoadMap &road_map,
                                                         const double max_sensor_range_m,
                                                         const int max_num_transforms,
                                                         const EkfSlam &ekf,
                                                         const BeaconPotential &beacon_potential,
                                                         const TransformType type) {
    std::unordered_map<DirectedEdge, EdgeTransform, DirectedEdgeHash> edge_transform_cache;
    return [&road_map, max_sensor_range_m, &ekf,
            edge_transform_cache = std::move(edge_transform_cache), &beacon_potential,
            max_num_transforms, type](const RobotBelief &initial_belief, const int start_idx,
                                      const int end_idx) mutable -> RobotBelief {
        // Get the belief edge transform, optionally updating the cache
        const DirectedEdge edge = {
            .source = start_idx,
            .destination = end_idx,
            .initial_heading_in_local = initial_belief.local_from_robot.so2().log(),
        };
        const auto cache_iter = edge_transform_cache.find(edge);
        const bool is_in_cache = cache_iter != edge_transform_cache.end();
        const auto end_pos_in_local = road_map.point(end_idx);
        const EdgeTransform &edge_transform =
            is_in_cache
                ? cache_iter->second
                : compute_edge_belief_transform(initial_belief.local_from_robot, end_pos_in_local,
                                                ekf.config(), ekf.estimate(), beacon_potential,
                                                max_sensor_range_m, max_num_transforms, type);
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
        if (type == TransformType::COVARIANCE) {
            input.topRightCorner(cov_dim, cov_dim) = initial_belief.cov_in_robot;
        } else {
            input.topRightCorner(cov_dim, cov_dim) = initial_belief.cov_in_robot.inverse();
        }

        Eigen::MatrixXd new_belief_in_robot = Eigen::MatrixXd::Zero(cov_dim, cov_dim);

        const double weight_total =
            std::accumulate(edge_transform.weight.begin(), edge_transform.weight.end(), 0.0);
        for (int i = 0; i < static_cast<int>(edge_transform.weight.size()); i++) {
            const double weight = edge_transform.weight.at(i) / weight_total;
            const auto &cov_transform = edge_transform.transforms.at(i);
            if (type == TransformType::COVARIANCE) {
                const ScatteringTransformBase result = math::redheffer_star(
                    input, std::get<ScatteringTransform<TransformType::COVARIANCE>>(cov_transform));
                const Eigen::MatrixXd sample_cov_in_robot = result.topRightCorner(cov_dim, cov_dim);
                new_belief_in_robot += weight * sample_cov_in_robot;
            } else {
                const ScatteringTransformBase result = math::redheffer_star(
                    input,
                    std::get<ScatteringTransform<TransformType::INFORMATION>>(cov_transform));
                const Eigen::MatrixXd sample_info_in_robot =
                    result.topRightCorner(cov_dim, cov_dim);
                new_belief_in_robot += weight * sample_info_in_robot;
            }
        }

        return RobotBelief{
            .local_from_robot = edge_transform.local_from_robot,
            .cov_in_robot = type == TransformType::COVARIANCE ? new_belief_in_robot
                                                              : new_belief_in_robot.inverse(),
        };
    };
}

planning::BeliefUpdater<RobotBelief> make_belief_updater(const planning::RoadMap &road_map,
                                                         const double max_sensor_range_m,
                                                         const EkfSlam &ekf,
                                                         const std::vector<int> &present_beacons,
                                                         const TransformType type) {
    std::unordered_map<DirectedEdge, std::tuple<liegroups::SE2, TypedTransform>, DirectedEdgeHash>
        edge_transform_cache;

    return [&road_map, max_sensor_range_m, &ekf,
            edge_transform_cache = std::move(edge_transform_cache), &present_beacons,
            type](const RobotBelief &initial_belief, const int start_idx,
                  const int end_idx) mutable -> RobotBelief {
        // Get the belief edge transform, optionally updating the cache
        const DirectedEdge edge = {
            .source = start_idx,
            .destination = end_idx,
            .initial_heading_in_local = initial_belief.local_from_robot.so2().log(),
        };
        const auto cache_iter = edge_transform_cache.find(edge);
        const bool is_in_cache = cache_iter != edge_transform_cache.end();
        const auto end_pos_in_local = road_map.point(end_idx);
        const auto &[local_from_new_robot, edge_transform] =
            is_in_cache ? cache_iter->second
                        : compute_edge_belief_transform(
                              initial_belief.local_from_robot, end_pos_in_local, ekf.config(),
                              ekf.estimate(), present_beacons, max_sensor_range_m, type);
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
        if (type == TransformType::COVARIANCE) {
            input.topRightCorner(cov_dim, cov_dim) = initial_belief.cov_in_robot;
        } else {
            input.topRightCorner(cov_dim, cov_dim) = initial_belief.cov_in_robot.inverse();
        }

        if (type == TransformType::COVARIANCE) {
            const ScatteringTransformBase result = math::redheffer_star(
                input, std::get<ScatteringTransform<TransformType::COVARIANCE>>(edge_transform));
            const Eigen::MatrixXd cov_in_robot = result.topRightCorner(cov_dim, cov_dim);
            return RobotBelief{
                .local_from_robot = local_from_new_robot,
                .cov_in_robot = cov_in_robot,
            };
        } else {
            const ScatteringTransformBase result = math::redheffer_star(
                input, std::get<ScatteringTransform<TransformType::INFORMATION>>(edge_transform));
            const Eigen::MatrixXd info_in_robot = result.topRightCorner(cov_dim, cov_dim);
            return RobotBelief{
                .local_from_robot = local_from_new_robot,
                .cov_in_robot = info_in_robot.inverse(),
            };
        }
    };
}

planning::BeliefUpdater<LandmarkRobotBelief> make_landmark_belief_updater(
    const planning::RoadMap &road_map, const double max_sensor_range_m, const EkfSlam &ekf,
    const BeaconPotential &beacon_potential, const TransformType type) {
    std::unordered_map<
        DirectedEdge,
        std::tuple<liegroups::SE2, std::vector<std::tuple<std::string, TypedTransform>>>,
        DirectedEdgeHash>
        edge_transform_cache;

    return [&road_map, max_sensor_range_m, &ekf,
            edge_transform_cache = std::move(edge_transform_cache), &beacon_potential,
            type](const LandmarkRobotBelief &initial_belief, const int start_idx,
                  const int end_idx) mutable -> LandmarkRobotBelief {
        (void)beacon_potential;
        // Get the belief edge transform, optionally updating the cache
        const DirectedEdge edge = {
            .source = start_idx,
            .destination = end_idx,
            .initial_heading_in_local = initial_belief.local_from_robot.so2().log(),
        };
        const auto cache_iter = edge_transform_cache.find(edge);
        const bool is_in_cache = cache_iter != edge_transform_cache.end();
        const auto end_pos_in_local = road_map.point(end_idx);
        const auto &[local_from_new_robot, edge_transform] =
            is_in_cache ? cache_iter->second
                        : compute_edge_belief_transform(
                              initial_belief.local_from_robot, end_pos_in_local, ekf.config(),
                              ekf.estimate(), beacon_potential, max_sensor_range_m, type);
        if (!is_in_cache) {
            // Add the transform to the cache in case it's missing
            edge_transform_cache[edge] = std::make_tuple(local_from_new_robot, edge_transform);
        }

        std::unordered_map<std::string, LandmarkRobotBelief::LandmarkConditionedRobotBelief>
            belief_from_config;

        for (const auto &[config, belief] : initial_belief.belief_from_config) {
            const auto consistent_transform_iterators =
                find_consistent_configs(config, edge_transform);

            for (const auto iter : consistent_transform_iterators) {
                const auto &[transform_config, transform] = *iter;

                const std::string new_config = merge_configurations(config, transform_config);

                //  Compute the new covariance
                //  [- new_cov] = [I cov] * edge_transform
                //  [-       -]   [0   I]
                //  new_cov  = A * B^-1
                const int cov_dim = belief.cov_in_robot.rows();
                Eigen::MatrixXd input = Eigen::MatrixXd::Identity(2 * cov_dim, 2 * cov_dim);
                if (type == TransformType::COVARIANCE) {
                    input.topRightCorner(cov_dim, cov_dim) = belief.cov_in_robot;
                } else {
                    input.topRightCorner(cov_dim, cov_dim) = belief.cov_in_robot.inverse();
                }

                // Compute the probability of this config
                constexpr bool ALLOW_PARTIAL_ASSIGNMENT = true;
                std::unordered_map<int, bool> assignment;

                for (int i = 0; i < static_cast<int>(new_config.size()); i++) {
                    if (new_config.at(i) == '?') {
                        continue;
                    } else if (new_config.at(i) == '1') {
                        assignment[beacon_potential.members().at(i)] = true;
                    } else if (new_config.at(i) == '0') {
                        assignment[beacon_potential.members().at(i)] = false;
                    } else {
                        CHECK(false, "unable to parse config string", new_config, i,
                              new_config.at(i));
                    }
                }

                const double log_config_prob =
                    beacon_potential.log_prob(assignment, ALLOW_PARTIAL_ASSIGNMENT);

                if (type == TransformType::COVARIANCE) {
                    const ScatteringTransformBase result = math::redheffer_star(
                        input, std::get<ScatteringTransform<TransformType::COVARIANCE>>(transform));
                    const Eigen::MatrixXd cov_in_robot = result.topRightCorner(cov_dim, cov_dim);
                    belief_from_config[new_config] = {
                        .cov_in_robot = cov_in_robot,
                        .log_config_prob = log_config_prob,
                    };
                } else {
                    const ScatteringTransformBase result = math::redheffer_star(
                        input,
                        std::get<ScatteringTransform<TransformType::INFORMATION>>(transform));
                    const Eigen::MatrixXd info_in_robot = result.topRightCorner(cov_dim, cov_dim);
                    belief_from_config[new_config] = {
                        .cov_in_robot = info_in_robot.inverse(),
                        .log_config_prob = log_config_prob,
                    };
                }
            }
        }

        return {
            .local_from_robot = local_from_new_robot,
            .belief_from_config = std::move(belief_from_config),
        };
    };
}

}  // namespace robot::experimental::beacon_sim
