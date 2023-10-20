
#include "experimental/beacon_sim/information_lower_bound_planner.hh"

#include <iostream>
#include <iterator>

#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/information_lower_bound_search.hh"
#include "experimental/beacon_sim/make_eigen_bounds.hh"

namespace robot::experimental::beacon_sim {
bool eigen_debug = false;
namespace {
std::vector<RobotBelief> compute_beliefs(
    const planning::RoadMap &road_map, const std::vector<std::tuple<int, double>> &path_and_bounds,
    const EkfSlam &ekf, const double max_sensor_range_m) {
    const auto belief_updater = make_belief_updater(
        road_map, max_sensor_range_m, ekf, ekf.estimate().beacon_ids, TransformType::COVARIANCE);

    std::cout << "plan: " << std::endl;
    std::vector<RobotBelief> out = {{.local_from_robot = ekf.estimate().local_from_robot(),
                                     .cov_in_robot = ekf.estimate().robot_cov()}};
    for (int i = 1; i < static_cast<int>(path_and_bounds.size()); i++) {
        const int start_idx = std::get<0>(path_and_bounds.at(i - 1));
        const auto &[end_idx, end_lower_bound] = path_and_bounds.at(i);
        out.push_back(belief_updater(out.back(), start_idx, end_idx));
        std::cout << "idx: " << i << " node id: " << end_idx << " end_lb: " << end_lower_bound
                  << out.back().cov_in_robot.inverse().eigenvalues().transpose() << std::endl;
    }
    std::cout << "end plan: " << std::endl;
    return out;
}
}  // namespace
planning::BRMPlan<RobotBelief> compute_info_lower_bound_plan(
    const planning::RoadMap &road_map, const EkfSlam &ekf,
    const double information_lower_bound_at_goal, const double max_sensor_range_m) {
    const Eigen::Matrix3d information = ekf.estimate().robot_cov().inverse();
    const double min_info_eigenvalue = information.eigenvalues().cwiseAbs().minCoeff();

    std::cout << "info_eigen_values: "
              << ekf.estimate().robot_cov().inverse().eigenvalues().transpose() << std::endl;
    std::cout << "starting info lower bound: " << min_info_eigenvalue << std::endl;
    std::cout << "goal info constraint: " << information_lower_bound_at_goal << std::endl;

    const LowerBoundReversePropagator reverse_propagator = [&ekf, max_sensor_range_m, &road_map](
                                                               const int start_idx,
                                                               const int end_idx,
                                                               const double lower_bound_at_end) {
        if (start_idx == 18 && end_idx == -2) {
            eigen_debug = true;
        }
        constexpr double local_from_robot_theta = 0.0;
        const Eigen::Vector2d start_in_local = road_map.point(start_idx);
        const liegroups::SE2 local_from_end_robot(local_from_robot_theta, road_map.point(end_idx));
        const Eigen::Vector2d start_in_end_robot = local_from_end_robot.inverse() * start_in_local;
        const double lower_bound_at_start = compute_backwards_eigen_bound_transform(
            lower_bound_at_end, local_from_end_robot, start_in_local, ekf.config(), ekf.estimate(),
            std::nullopt, max_sensor_range_m);

        if (eigen_debug) {
            eigen_debug = false;
        }

        return PropagationResult{.info_lower_bound = lower_bound_at_start,
                                 .edge_cost = start_in_end_robot.norm()};
    };

    const auto result = information_lower_bound_search(
        road_map, min_info_eigenvalue, information_lower_bound_at_goal, reverse_propagator);

    std::cout << "info lower bound at start: " << result.info_lower_bound << std::endl;

    std::vector<int> nodes;
    std::transform(result.path_and_bounds_to_goal.begin(), result.path_and_bounds_to_goal.end(),
                   std::back_inserter(nodes), [](const auto &node_and_bound) {
                       const auto &[node_id, _] = node_and_bound;
                       return node_id;
                   });

    return {
        .nodes = nodes,
        .beliefs =
            compute_beliefs(road_map, result.path_and_bounds_to_goal, ekf, max_sensor_range_m),
    };
}
}  // namespace robot::experimental::beacon_sim
