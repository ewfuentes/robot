
#pragma once

#include <optional>
#include <unordered_set>
#include <variant>

#include "Eigen/Core"
#include "common/liegroups/se2.hh"
#include "common/math/redheffer_star.hh"
#include "experimental/beacon_sim/correlated_beacons.hh"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/robot_belief.hh"
#include "planning/belief_road_map.hh"
#include "planning/probabilistic_road_map.hh"

namespace robot::experimental::beacon_sim {

enum class TransformType {
    INFORMATION,
    COVARIANCE,
};

using ScatteringTransformBase =
    Eigen::Matrix<double, 2 * liegroups::SE2::DoF, 2 * liegroups::SE2::DoF>;
template <TransformType T>
struct ScatteringTransform : ScatteringTransformBase {
    static constexpr TransformType TYPE = T;

    ScatteringTransform(void) : ScatteringTransformBase() {}

    template <typename OtherDerived>
    ScatteringTransform(const Eigen::MatrixBase<OtherDerived> &other)
        : ScatteringTransformBase(other) {}

    template <typename OtherDerived>
    ScatteringTransform &operator=(const Eigen::MatrixBase<OtherDerived> &other) {
        this->ScatteringTransformBase::operator=(other);
        return *this;
    }

    static ScatteringTransform Identity() {
        return ScatteringTransform(ScatteringTransformBase::Identity());
    }
};

using TypedTransform = std::variant<ScatteringTransform<TransformType::COVARIANCE>,
                                    ScatteringTransform<TransformType::INFORMATION>>;
using TypedTransformVector =
    std::vector<std::variant<ScatteringTransform<TransformType::COVARIANCE>,
                             ScatteringTransform<TransformType::INFORMATION>>>;

template <TransformType T>
ScatteringTransform<T> operator*(const ScatteringTransform<T> &a, const ScatteringTransform<T> &b) {
    return math::redheffer_star(a, b);
}

std::optional<TypedTransform> operator*(const TypedTransform &a, const TypedTransform &b);

struct EdgeTransform {
    liegroups::SE2 local_from_robot;
    std::vector<double> weight;
    TypedTransformVector transforms;
};
Eigen::DiagonalMatrix<double, 3> compute_process_noise(const EkfSlamConfig &config,
                                                       const double dt_s, const double arclength_m);
TypedTransform compute_process_transform(const Eigen::Matrix3d &process_noise,
                                         const liegroups::SE2 &old_robot_from_new_robot,
                                         const TransformType type);
TypedTransform compute_measurement_transform(
    const liegroups::SE2 &local_from_robot, const EkfSlamConfig &ekf_config,
    const EkfSlamEstimate &ekf_estimate, const std::optional<std::vector<int>> &available_beacons,
    const double max_sensor_range_m, const TransformType type);

Eigen::MatrixXd build_measurement_noise(const int num_observations,
                                        const double range_measurement_noise_m,
                                        const double bearing_measurement_noise_rad);

std::tuple<liegroups::SE2, TypedTransform> compute_edge_belief_transform(
    const liegroups::SE2 &local_from_robot, const Eigen::Vector2d &end_state_in_local,
    const EkfSlamConfig &ekf_config, const EkfSlamEstimate &ekf_estimate,
    const std::optional<std::vector<int>> &available_beacons, const double max_sensor_range_m,
    const TransformType transform_type);

EdgeTransform compute_edge_belief_transform(
    const liegroups::SE2 &local_from_robot, const Eigen::Vector2d &end_state_in_local,
    const EkfSlamConfig &ekf_config, const EkfSlamEstimate &ekf_estimate,
    const BeaconPotential &beacon_potential, const double max_sensor_range_m,
    const int max_num_transforms, const TransformType transform_type);

planning::BeliefUpdater<RobotBelief> make_belief_updater(const planning::RoadMap &road_map,
                                                         const double max_sensor_range_m,
                                                         const int max_num_transforms,
                                                         const EkfSlam &ekf,
                                                         const BeaconPotential &beacon_potential,
                                                         const TransformType transform_type);

planning::BeliefUpdater<RobotBelief> make_belief_updater(const planning::RoadMap &road_map,
                                                         const double max_sensor_range_m,
                                                         const EkfSlam &ekf,
                                                         const std::vector<int> &present_beacons,
                                                         const TransformType transform_type);

}  // namespace robot::experimental::beacon_sim
