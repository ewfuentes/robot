
#include "experimental/beacon_sim/belief_road_map_planner.hh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "Eigen/Core"
#include "common/liegroups/se2.hh"
#include "common/math/combinations.hh"
#include "common/math/redheffer_star.hh"
#include "experimental/beacon_sim/correlated_beacons.hh"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/generate_observations.hh"
#include "experimental/beacon_sim/make_belief_updater.hh"
#include "experimental/beacon_sim/robot.hh"
#include "planning/belief_road_map.hh"
#include "planning/breadth_first_search.hh"
#include "planning/probabilistic_road_map.hh"

namespace robot::experimental::beacon_sim {
namespace {

std::vector<std::vector<int>> find_paths(const Eigen::Vector2d start_in_local,
                                         const Eigen::Vector2d goal_in_local,
                                         const planning::RoadMap &road_map,
                                         [[maybe_unused]] const double max_path_length_ratio,
                                         const double start_goal_connection_radius_m) {
    constexpr int START_IDX = -1;
    constexpr int GOAL_IDX = -2;

    const planning::SuccessorFunc<int> successors_func =
        [&road_map, start_goal_connection_radius_m, start_in_local,
         goal_in_local](const int &node_idx) -> std::vector<planning::Successor<int>> {
        if (node_idx == GOAL_IDX) {
            return {};
        }

        std::vector<planning::Successor<int>> out;
        if (node_idx == START_IDX) {
            for (int i = 0; i < static_cast<int>(road_map.points.size()); i++) {
                const Eigen::Vector2d &pt_in_local = road_map.points.at(i);
                const double dist_m = (pt_in_local - start_in_local).norm();
                if (dist_m < start_goal_connection_radius_m) {
                    out.push_back({.state = i, .edge_cost = dist_m});
                }
            }
        } else {
            const Eigen::Vector2d &curr_pt_in_local = road_map.points.at(node_idx);
            for (int i = 0; i < static_cast<int>(road_map.points.size()); i++) {
                if (road_map.adj(i, node_idx)) {
                    const Eigen::Vector2d &other_in_local = road_map.points.at(i);
                    const double dist_m = (curr_pt_in_local - other_in_local).norm();
                    out.push_back({.state = i, .edge_cost = dist_m});
                }
            }

            const double dist_to_goal_m = (curr_pt_in_local - goal_in_local).norm();
            if (dist_to_goal_m < start_goal_connection_radius_m) {
                out.push_back({.state = GOAL_IDX, .edge_cost = dist_to_goal_m});
            }
        }
        return out;
    };

    std::optional<double> shortest_path_length = std::nullopt;
    std::vector<std::vector<int>> out;
    const planning::ShouldQueueFunc<int> should_queue_func =
        [&shortest_path_length, max_path_length_ratio, &road_map, &goal_in_local, &out](
            const planning::Successor<int> &successor, const int parent_idx,
            const std::vector<planning::Node<int>> &node_list) mutable {
            // Compute the cost so far
            const double path_length_m = node_list.at(parent_idx).cost + successor.edge_cost;
            // Compute a lower bound on distance to goal
            const double estimated_dist_to_goal_m = [&]() {
                if (successor.state == GOAL_IDX) {
                    return 0.0;
                }
                const Eigen::Vector2d &pt_in_local = road_map.points.at(successor.state);
                return (goal_in_local - pt_in_local).norm();
            }();

            const double threshold_m = shortest_path_length.has_value()
                                           ? (shortest_path_length.value() * max_path_length_ratio)
                                           : std::numeric_limits<double>::max();
            if (path_length_m + estimated_dist_to_goal_m > threshold_m) {
                return planning::ShouldQueueResult::SKIP;
            }

            if (successor.state == GOAL_IDX) {
                if (!shortest_path_length.has_value() ||
                    path_length_m < shortest_path_length.value()) {
                    shortest_path_length = path_length_m;
                }
                // collect the path
                std::vector<int> new_path = {GOAL_IDX};
                std::optional<int> maybe_idx = parent_idx;
                while (maybe_idx.has_value()) {
                    const planning::Node<int> &n = node_list.at(maybe_idx.value());
                    new_path.push_back(n.state);
                    maybe_idx = n.maybe_parent_idx;
                }
                std::reverse(new_path.begin(), new_path.end());
                out.emplace_back(std::move(new_path));
            }
            return planning::ShouldQueueResult::QUEUE;
        };

    const planning::GoalCheckFunc<int> goal_check_func = [](const planning::Node<int> &) {
        return false;
    };

    const planning::IdentifyPathEndFunc<int> identify_end_func = [](const auto &) {
        return std::nullopt;
    };

    planning::breadth_first_search(START_IDX, successors_func, should_queue_func, goal_check_func,
                                   identify_end_func);

    return out;
}

Eigen::Matrix3d evaluate_path(const std::vector<int> &path, const RobotBelief &initial_belief,
                              const planning::BeliefUpdater<RobotBelief> &updater) {
    RobotBelief robot_belief = initial_belief;
    for (int i = 1; i < static_cast<int>(path.size()); i++) {
        robot_belief = updater(robot_belief, path.at(i - 1), path.at(i));
    }

    return robot_belief.cov_in_robot;
}

std::vector<Eigen::Matrix3d> evaluate_paths_with_configuration(
    const std::vector<std::vector<int>> &paths, const Eigen::Vector2d &goal_in_local,
    const EkfSlam &ekf, const planning::RoadMap &road_map, const double max_sensor_range_m,
    const std::vector<int> &present_beacons) {
    // Make a belief updater that only considers the present beacons
    const auto updater = make_belief_updater(road_map, goal_in_local, max_sensor_range_m, ekf,
                                             present_beacons, TransformType::COVARIANCE);

    const RobotBelief initial_belief = {
        .local_from_robot = ekf.estimate().local_from_robot(),
        .cov_in_robot = ekf.estimate().robot_cov(),
    };

    std::vector<Eigen::Matrix3d> out;
    std::transform(
        paths.begin(), paths.end(), std::back_inserter(out),
        [&](const std::vector<int> &path) { return evaluate_path(path, initial_belief, updater); });
    return out;
}

std::vector<Eigen::Matrix3d> evaluate_paths(std::vector<std::vector<int>> &paths,
                                            const Eigen::Vector2d &goal_in_local,
                                            const EkfSlam &ekf, const planning::RoadMap &road_map,
                                            const double max_sensor_range_m,
                                            const BeaconPotential &potential) {
    std::vector<Eigen::Matrix3d> expected_covs(paths.size(), Eigen::Matrix3d::Zero());

    double weight_sum = 0;
    for (int i = 0; i <= static_cast<int>(potential.members().size()); i++) {
        for (const std::vector<int> &present_idxs :
             math::combinations(potential.members().size(), i)) {
            const std::vector<int> present_beacons = [&]() {
                std::vector<int> out;
                for (int idx : present_idxs) {
                    out.push_back(potential.members().at(idx));
                }
                return out;
            }();

            // Compute the probability of the configuration
            const double p = std::exp(potential.log_prob(present_beacons));
            weight_sum += p;

            // Compute the covariance for each path
            std::vector<Eigen::Matrix3d> covs_for_config = evaluate_paths_with_configuration(
                paths, goal_in_local, ekf, road_map, max_sensor_range_m, present_beacons);

            for (int i = 0; i < static_cast<int>(covs_for_config.size()); i++) {
                expected_covs.at(i) += p * covs_for_config.at(i);
            }
        }
    }

    for (int i = 0; i < static_cast<int>(expected_covs.size()); i++) {
        expected_covs.at(i) /= weight_sum;
    }

    return expected_covs;
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
    const Eigen::Vector2d &goal_state, const BeliefRoadMapOptions &options) {
    const auto &estimate = ekf.estimate();

    const RobotBelief initial_belief = {
        .local_from_robot = estimate.local_from_robot(),
        .cov_in_robot = estimate.robot_cov(),
    };
    const auto belief_updater = make_belief_updater(
        road_map, goal_state, options.max_sensor_range_m, options.max_num_edge_transforms, ekf,
        beacon_potential, TransformType::COVARIANCE);
    if (options.uncertainty_tolerance.has_value()) {
        return planning::plan<RobotBelief>(
            road_map, initial_belief, belief_updater, goal_state, options.num_start_connections,
            options.num_goal_connections,
            planning::MinUncertaintyToleranceOptions{options.uncertainty_tolerance.value()});
    } else {
        return planning::plan<RobotBelief>(
            road_map, initial_belief, belief_updater, goal_state, options.num_start_connections,
            options.num_goal_connections, planning::NoBacktrackingOptions{});
    }
}

ExpectedBeliefPlanResult compute_expected_belief_road_map_plan(
    const planning::RoadMap &road_map, const EkfSlam &ekf, const BeaconPotential &beacon_potential,
    const Eigen::Vector2d &goal_state, const ExpectedBeliefRoadMapOptions &options) {
    // Find paths that are at most options.max_path_length_ratio times longer than the shortest path
    std::vector<std::vector<int>> paths =
        find_paths(ekf.estimate().local_from_robot().translation(), goal_state, road_map,
                   options.max_path_length_ratio, options.start_goal_connection_radius_m);

    // Now that we have the expected paths, evaluate each path across all configurations

    const std::vector<Eigen::Matrix3d> expected_covs = evaluate_paths(
        paths, goal_state, ekf, road_map, options.max_sensor_range_m, beacon_potential);

    for (int i = 0; i < static_cast<int>(paths.size()); i++) {
        std::cout << "[";
        for (const int node : paths.at(i)) {
            std::cout << node << " ";
        }
        [[maybe_unused]] const double det = expected_covs.at(i).determinant();
        std::cout << "] " << det << std::endl;
    }

    std::cout << "Found " << paths.size() << " paths to goal" << std::endl;
    const auto iter = std::min_element(expected_covs.begin(), expected_covs.end(),
                                       [](const Eigen::Matrix3d &a, const Eigen::Matrix3d &b) {
                                           return a.determinant() < b.determinant();
                                       });
    const int idx = std::distance(expected_covs.begin(), iter);

    return ExpectedBeliefPlanResult{
        .plan = paths.at(idx),
        .expected_cov = expected_covs.at(idx),
    };
}

namespace detail {}  // namespace detail
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
