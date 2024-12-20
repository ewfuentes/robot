
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
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

#include "Eigen/Core"
#include "common/check.hh"
#include "common/liegroups/se2.hh"
#include "common/math/combinations.hh"
#include "common/math/logsumexp.hh"
#include "common/math/multivariate_normal_cdf.hh"
#include "common/math/redheffer_star.hh"
#include "common/time/robot_time.hh"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/generate_observations.hh"
#include "experimental/beacon_sim/make_belief_updater.hh"
#include "experimental/beacon_sim/robot.hh"
#include "experimental/beacon_sim/robot_belief.hh"
#include "planning/belief_road_map.hh"
#include "planning/breadth_first_search.hh"
#include "planning/probabilistic_road_map.hh"
#include "planning/road_map.hh"

namespace robot::experimental::beacon_sim {
namespace {
std::vector<std::vector<int>> find_paths(const planning::RoadMap &road_map,
                                         const double max_path_length_ratio) {
    const planning::SuccessorFunc<int> successors_func =
        [&road_map](const int &node_idx) -> std::vector<planning::Successor<int>> {
        if (node_idx == planning::RoadMap::GOAL_IDX) {
            return {};
        }

        std::vector<planning::Successor<int>> out;
        const Eigen::Vector2d curr_pt_in_local = road_map.point(node_idx);
        for (const auto &[other_node_id, other_in_local] : road_map.neighbors(node_idx)) {
            const double dist_m = (curr_pt_in_local - other_in_local).norm();
            out.push_back({.state = other_node_id, .edge_cost = dist_m});
        }
        return out;
    };

    std::optional<double> shortest_path_length = std::nullopt;
    std::vector<std::vector<int>> out;
    const planning::ShouldQueueFunc<int> should_queue_func =
        [&shortest_path_length, max_path_length_ratio, &road_map, &out](
            const planning::Successor<int> &successor, const int parent_idx,
            const std::vector<planning::Node<int>> &node_list) mutable {
            // Compute the cost so far
            const double path_length_m = node_list.at(parent_idx).cost + successor.edge_cost;
            // Compute a lower bound on distance to goal
            const double estimated_dist_to_goal_m = [&]() {
                if (successor.state == planning::RoadMap::GOAL_IDX) {
                    return 0.0;
                }
                const Eigen::Vector2d &pt_in_local = road_map.point(successor.state);
                return (road_map.point(planning::RoadMap::GOAL_IDX) - pt_in_local).norm();
            }();

            const double threshold_m = shortest_path_length.has_value()
                                           ? (shortest_path_length.value() * max_path_length_ratio)
                                           : std::numeric_limits<double>::max();
            if (path_length_m + estimated_dist_to_goal_m > threshold_m) {
                return planning::ShouldQueueResult::SKIP;
            }

            if (successor.state == planning::RoadMap::GOAL_IDX) {
                if (!shortest_path_length.has_value() ||
                    path_length_m < shortest_path_length.value()) {
                    shortest_path_length = path_length_m;
                }
                // collect the path
                std::vector<int> new_path = {planning::RoadMap::GOAL_IDX};
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

    planning::breadth_first_search(planning::RoadMap::START_IDX, successors_func, should_queue_func,
                                   goal_check_func, identify_end_func);

    return out;
}

RobotBelief evaluate_path(const std::vector<int> &path, const RobotBelief &initial_belief,
                          const planning::BeliefUpdater<RobotBelief> &updater) {
    RobotBelief robot_belief = initial_belief;
    for (int i = 1; i < static_cast<int>(path.size()); i++) {
        robot_belief = updater(robot_belief, path.at(i - 1), path.at(i));
    }

    return robot_belief;
}

std::vector<Eigen::Matrix3d> evaluate_paths(std::vector<std::vector<int>> &paths,
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
            std::vector<RobotBelief> beliefs_for_config = evaluate_paths_with_configuration(
                paths, ekf, road_map, max_sensor_range_m, present_beacons);

            for (int i = 0; i < static_cast<int>(beliefs_for_config.size()); i++) {
                expected_covs.at(i) += p * beliefs_for_config.at(i).cov_in_robot;
            }
        }
    }

    for (int i = 0; i < static_cast<int>(expected_covs.size()); i++) {
        expected_covs.at(i) /= weight_sum;
    }

    return expected_covs;
}
}  // namespace

std::vector<RobotBelief> evaluate_paths_with_configuration(
    const std::vector<std::vector<int>> &paths, const EkfSlam &ekf,
    const planning::RoadMap &road_map, const double max_sensor_range_m,
    const std::vector<int> &present_beacons) {
    // Make a belief updater that only considers the present beacons
    const auto updater = make_belief_updater(road_map, max_sensor_range_m, ekf, present_beacons,
                                             TransformType::COVARIANCE);

    const RobotBelief initial_belief = {
        .local_from_robot = ekf.estimate().local_from_robot(),
        .cov_in_robot = ekf.estimate().robot_cov(),
    };

    std::vector<RobotBelief> out;
    std::transform(
        paths.begin(), paths.end(), std::back_inserter(out),
        [&](const std::vector<int> &path) { return evaluate_path(path, initial_belief, updater); });
    return out;
}

template <typename T>
std::function<double(const T &)> make_uncertainty_size(const UncertaintySizeOptions &);

template <>
std::function<double(const RobotBelief &)> make_uncertainty_size(
    const UncertaintySizeOptions &options) {
    if (std::holds_alternative<ValueAtRiskDeterminant>(options)) {
        ROBOT_CHECK(false, "ValueAtRiskDeterminant is not supported for unimodal beliefs", options);
    } else if (std::holds_alternative<ExpectedDeterminant>(options)) {
        return [opt = std::get<ExpectedDeterminant>(options)](const RobotBelief &belief) -> double {
            if (opt.position_only) {
                return belief.cov_in_robot.topLeftCorner<2, 2>().determinant();
            } else {
                return belief.cov_in_robot.determinant();
            };
        };
    } else if (std::holds_alternative<ExpectedTrace>(options)) {
        return [opt = std::get<ExpectedTrace>(options)](const RobotBelief &belief) -> double {
            if (opt.position_only) {
                return belief.cov_in_robot.topLeftCorner<2, 2>().trace();
            } else {
                return belief.cov_in_robot.trace();
            };
        };
    } else if (std::holds_alternative<ProbMassInRegion>(options)) {
        return [opt = std::get<ProbMassInRegion>(options)](const RobotBelief &belief) -> double {
            const Eigen::Vector3d bounds =
                Eigen::Vector3d{opt.position_x_half_width_m, opt.position_y_half_width_m,
                                opt.heading_half_width_rad};

            const auto compute_component_mass = [&bounds](const Eigen::Matrix3d &cov) {
                // clang-format off
                // The quadrants are numbered in the usual way, but the negative Z direction
                // adds +4 to the index. So the 0th octant is the space spanned by postive
                // linear combinations of (+x, +y, +z) and the 4th octant is the space 
                // spanned by (+x, +y, -z). Note that +Z is out of the screen.
                //
                //           +y
                //           ▲
                //           │
                //      1/5  │   0/4
                //           │
                //           │
                //   ◄───────0────────► +x
                //           │ +z
                //           │
                //      2/6  │   3/7
                //           │
                //           ▼
                //
                // Imagine that we shift the origin such it is colocated with the point in the
                // 6th octant (spanned by (-x, -y, -z)). Computing the probability of the region
                // of interest requires computing the probability mass that exists in octant 0.
                //
                // However, we only have access to the CDF. Let P(x) be the value of the CDF when
                // evaluated at the xth region point in a manner consistent with the octant 
                // numbering. Then:
                // P(0) contains octants 0, 1, 2, 3, 4, 5, 6, 7
                // P(1) contains octants    1, 2,       5, 6
                // P(2) contains octants       2,          6
                // P(3) contains octants       2, 3,       6, 7
                // P(4) contains octants             4, 5, 6, 7
                // P(5) contains octants                5, 6
                // P(6) contains octants                   6
                // P(7) contains octants                   6, 7
                //
                // We want a linear combination of these values such that only 0 remains.
                // This can be achieved by:
                // P(0) - P(1) + P(2) - P(3) - P(4)  + P(5) - P(6) + P(7)
                // 
                // These are the corner points of a rectangular prism in octant order
                const std::vector<Eigen::Vector3d> eval_pts = {
                    {bounds.x(), bounds.y(), bounds.z()},
                    {-bounds.x(), bounds.y(), bounds.z()},   
                    {-bounds.x(), -bounds.y(), bounds.z()},  
                    {bounds.x(), -bounds.y(), bounds.z()},
                    {bounds.x(), bounds.y(), -bounds.z()},
                    {-bounds.x(), bounds.y(), -bounds.z()}, 
                    {-bounds.x(), -bounds.y(), -bounds.z()},
                    {bounds.x(), -bounds.y(), -bounds.z()},
                };
                // clang-format on

                std::vector<double> probs;
                for (const auto &eval_pt : eval_pts) {
                    const auto result =
                        math::multivariate_normal_cdf(Eigen::Vector3d::Zero(), cov, eval_pt);
                    ROBOT_CHECK(result.has_value(), "Could not compute cdf at eval pt", eval_pt);
                    probs.push_back(result.value());
                }
                return probs.at(0) - probs.at(1) + probs.at(2) - probs.at(3) - probs.at(4) +
                       probs.at(5) - probs.at(6) + probs.at(7);
            };

            double prob = compute_component_mass(belief.cov_in_robot);
            return 1 - prob;
        };
    }

    ROBOT_CHECK(false, "Unknown Uncertainty Size Options", options);
    return [](const auto &) { return 0.0; };
}

template <>
std::function<double(const LandmarkRobotBelief &)> make_uncertainty_size(
    const UncertaintySizeOptions &options) {
    if (std::holds_alternative<ValueAtRiskDeterminant>(options)) {
        return [opt = std::get<ValueAtRiskDeterminant>(options)](
                   const LandmarkRobotBelief &belief) -> double {
            std::vector<LandmarkRobotBelief::LandmarkConditionedRobotBelief> elements;
            elements.reserve(belief.belief_from_config.size());
            for (const auto &[_, value] : belief.belief_from_config) {
                elements.push_back(value);
            }
            std::sort(elements.begin(), elements.end(), [](const auto &a, const auto &b) {
                return a.cov_in_robot.determinant() < b.cov_in_robot.determinant();
            });

            double accumulated_prob = 0.0;
            for (const auto &elem : elements) {
                accumulated_prob += std::exp(elem.log_config_prob);
                if (accumulated_prob > opt.percentile) {
                    return elem.cov_in_robot.determinant();
                }
            }
            ROBOT_CHECK(false,
                        "Landmark Belief has insufficient probability mass to get to threshold",
                        accumulated_prob);
            return elements.back().cov_in_robot.determinant();
        };
    } else if (std::holds_alternative<ExpectedDeterminant>(options)) {
        return [opt = std::get<ExpectedDeterminant>(options)](
                   const LandmarkRobotBelief &belief) -> double {
            double expected_det = 0.0;
            for (const auto &[key, component] : belief.belief_from_config) {
                if (opt.position_only) {
                    const Eigen::Matrix2d &pos_covariance =
                        component.cov_in_robot.topLeftCorner<2, 2>();
                    expected_det +=
                        std::exp(component.log_config_prob) * pos_covariance.determinant();
                } else {
                    expected_det +=
                        std::exp(component.log_config_prob) * component.cov_in_robot.determinant();
                }
            }
            return expected_det;
        };
    } else if (std::holds_alternative<ExpectedTrace>(options)) {
        return [opt =
                    std::get<ExpectedTrace>(options)](const LandmarkRobotBelief &belief) -> double {
            double expected_trace = 0.0;
            for (const auto &[key, component] : belief.belief_from_config) {
                if (opt.position_only) {
                    const Eigen::Matrix2d &pos_covariance =
                        component.cov_in_robot.topLeftCorner<2, 2>();
                    expected_trace += std::exp(component.log_config_prob) * pos_covariance.trace();
                } else {
                    expected_trace +=
                        std::exp(component.log_config_prob) * component.cov_in_robot.trace();
                }
            }
            return expected_trace;
        };
    } else if (std::holds_alternative<ProbMassInRegion>(options)) {
        return [opt = std::get<ProbMassInRegion>(options)](
                   const LandmarkRobotBelief &belief) -> double {
            const Eigen::Vector3d bounds =
                Eigen::Vector3d{opt.position_x_half_width_m, opt.position_y_half_width_m,
                                opt.heading_half_width_rad};

            const auto compute_component_mass = [&bounds](const Eigen::Matrix3d &cov) {
                // clang-format off
                // The quadrants are numbered in the usual way, but the negative Z direction
                // adds +4 to the index. So the 0th octant is the space spanned by postive
                // linear combinations of (+x, +y, +z) and the 4th octant is the space 
                // spanned by (+x, +y, -z). Note that +Z is out of the screen.
                //
                //           +y
                //           ▲
                //           │
                //      1/5  │   0/4
                //           │
                //           │
                //   ◄───────0────────► +x
                //           │ +z
                //           │
                //      2/6  │   3/7
                //           │
                //           ▼
                //
                // Imagine that we shift the origin such it is colocated with the point in the
                // 6th octant (spanned by (-x, -y, -z)). Computing the probability of the region
                // of interest requires computing the probability mass that exists in octant 0.
                //
                // However, we only have access to the CDF. Let P(x) be the value of the CDF when
                // evaluated at the xth region point in a manner consistent with the octant 
                // numbering. Then:
                // P(0) contains octants 0, 1, 2, 3, 4, 5, 6, 7
                // P(1) contains octants    1, 2,       5, 6
                // P(2) contains octants       2,          6
                // P(3) contains octants       2, 3,       6, 7
                // P(4) contains octants             4, 5, 6, 7
                // P(5) contains octants                5, 6
                // P(6) contains octants                   6
                // P(7) contains octants                   6, 7
                //
                // We want a linear combination of these values such that only 0 remains.
                // This can be achieved by:
                // P(0) - P(1) + P(2) - P(3) - P(4)  + P(5) - P(6) + P(7)
                // 
                // These are the corner points of a rectangular prism in octant order
                const std::vector<Eigen::Vector3d> eval_pts = {
                    {bounds.x(), bounds.y(), bounds.z()},
                    {-bounds.x(), bounds.y(), bounds.z()},   
                    {-bounds.x(), -bounds.y(), bounds.z()},  
                    {bounds.x(), -bounds.y(), bounds.z()},
                    {bounds.x(), bounds.y(), -bounds.z()},
                    {-bounds.x(), bounds.y(), -bounds.z()}, 
                    {-bounds.x(), -bounds.y(), -bounds.z()},
                    {bounds.x(), -bounds.y(), -bounds.z()},
                };
                // clang-format on

                std::vector<double> probs;
                for (const auto &eval_pt : eval_pts) {
                    const auto result =
                        math::multivariate_normal_cdf(Eigen::Vector3d::Zero(), cov, eval_pt);
                    ROBOT_CHECK(result.has_value(), "Could not compute cdf at eval pt", eval_pt);
                    probs.push_back(result.value());
                }
                return probs.at(0) - probs.at(1) + probs.at(2) - probs.at(3) - probs.at(4) +
                       probs.at(5) - probs.at(6) + probs.at(7);
            };

            double prob = 0.0;
            for (const auto &[_, component] : belief.belief_from_config) {
                prob += std::exp(component.log_config_prob) *
                        compute_component_mass(component.cov_in_robot);
            }
            return 1 - prob;
        };
    }

    ROBOT_CHECK(false, "Unknown Uncertainty Size Options", options);
    return [](const auto &) { return 0.0; };
}

bool operator==(const RobotBelief &a, const RobotBelief &b) {
    constexpr double TOL = 1e-3;
    // Note that we don't consider covariance
    const auto mean_diff =
        (a.local_from_robot.translation() - b.local_from_robot.translation()).norm();

    const bool is_mean_near = mean_diff < TOL;
    return is_mean_near;
}

bool operator==(const LandmarkRobotBelief &a, const LandmarkRobotBelief &b) {
    constexpr double TOL = 1e-3;
    // Note that we don't consider covariance
    const auto mean_diff =
        (a.local_from_robot.translation() - b.local_from_robot.translation()).norm();

    const bool is_mean_near = mean_diff < TOL;
    return is_mean_near;
}

std::optional<planning::BRMPlan<RobotBelief>> compute_belief_road_map_plan(
    const planning::RoadMap &road_map, const EkfSlam &ekf, const BeaconPotential &beacon_potential,
    const BeliefRoadMapOptions &options) {
    const auto &estimate = ekf.estimate();

    const RobotBelief initial_belief = {
        .local_from_robot = estimate.local_from_robot(),
        .cov_in_robot = estimate.robot_cov(),
    };
    const auto belief_updater =
        make_belief_updater(road_map, options.max_sensor_range_m, options.max_num_edge_transforms,
                            ekf, beacon_potential, TransformType::COVARIANCE);

    const time::RobotTimestamp plan_start_time = time::current_robot_time();
    const auto should_terminate_func = [&]() {
        const auto elapsed_time = time::current_robot_time() - plan_start_time;
        const bool should_bail =
            options.timeout.has_value() && options.timeout.value() < elapsed_time;
        return should_bail;
    };

    const auto uncertainty_size =
        make_uncertainty_size<RobotBelief>(options.uncertainty_size_options);

    if (options.uncertainty_tolerance.has_value()) {
        return planning::plan(
            road_map, initial_belief, belief_updater, uncertainty_size,
            planning::MinUncertaintyToleranceOptions{options.uncertainty_tolerance.value()},
            should_terminate_func);
    } else {
        return planning::plan(road_map, initial_belief, belief_updater, uncertainty_size,
                              planning::NoBacktrackingOptions{}, should_terminate_func);
    }
}

PathConstrainedBeliefPlanResult compute_path_constrained_belief_road_map_plan(
    const planning::RoadMap &road_map, const EkfSlam &ekf, const BeaconPotential &beacon_potential,
    const PathConstrainedBeliefRoadMapOptions &options) {
    // Find paths that are at most options.max_path_length_ratio times longer than the shortest path
    std::vector<std::vector<int>> paths = find_paths(road_map, options.max_path_length_ratio);

    // Now that we have the expected paths, evaluate each path across all configurations

    const std::vector<Eigen::Matrix3d> expected_covs =
        evaluate_paths(paths, ekf, road_map, options.max_sensor_range_m, beacon_potential);

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

    return PathConstrainedBeliefPlanResult{
        .plan = paths.at(idx),
        .expected_cov = expected_covs.at(idx),
    };
}

std::optional<planning::BRMPlan<LandmarkRobotBelief>> compute_landmark_belief_road_map_plan(
    const planning::RoadMap &road_map, const EkfSlam &ekf, const BeaconPotential &beacon_potential,
    const LandmarkBeliefRoadMapOptions &options) {
    const auto &estimate = ekf.estimate();

    const LandmarkRobotBelief initial_belief = {
        .local_from_robot = estimate.local_from_robot(),
        .log_probability_mass_tracked = 0,
        .belief_from_config = {{std::string(beacon_potential.members().size(), '?'),
                                {.cov_in_robot = estimate.robot_cov(), .log_config_prob = 0}}}};
    const std::optional<SampledLandmarkBeliefOptions> sampled_belief_options =
        options.sampled_belief_options.has_value()
            ? std::make_optional<SampledLandmarkBeliefOptions>(
                  {.max_num_components = options.sampled_belief_options->max_num_components,
                   .seed = options.sampled_belief_options->seed})
            : std::nullopt;
    const auto belief_updater =
        make_landmark_belief_updater(road_map, options.max_sensor_range_m, sampled_belief_options,
                                     ekf, beacon_potential, TransformType::COVARIANCE);

    const time::RobotTimestamp plan_start_time = time::current_robot_time();
    const auto should_terminate_func = [&]() {
        const auto elapsed_time = time::current_robot_time() - plan_start_time;
        const bool should_bail =
            options.timeout.has_value() && options.timeout.value() < elapsed_time;
        return should_bail;
    };

    return planning::plan(
        road_map, initial_belief, belief_updater,
        make_uncertainty_size<LandmarkRobotBelief>(options.uncertainty_size_options),
        planning::NoBacktrackingOptions{}, should_terminate_func);
}

std::optional<ExpectedBeliefPlanResult> compute_expected_belief_road_map_plan(
    const planning::RoadMap &road_map, const EkfSlam &ekf, const BeaconPotential &beacon_potential,
    const ExpectedBeliefRoadMapOptions &options) {
    std::mt19937 gen(options.seed);
    const time::RobotTimestamp plan_start_time = time::current_robot_time();

    // Sample a number of worlds and run the BRM on it
    std::vector<std::vector<int>> world_samples;
    std::vector<std::vector<int>> plans;
    std::vector<double> log_probs;
    world_samples.reserve(options.num_configuration_samples);
    for (int i = 0; i < static_cast<int>(options.num_configuration_samples); i++) {
        world_samples.emplace_back(beacon_potential.sample(make_in_out(gen)));
        const auto &sample = world_samples.back();
        log_probs.push_back(beacon_potential.log_prob(sample));

        // Create a potential with only this assignment
        std::unordered_map<int, bool> assignment;
        for (const int beacon_id : beacon_potential.members()) {
            const auto iter = std::find(sample.begin(), sample.end(), beacon_id);
            const bool is_beacon_present_in_sample = iter != sample.end();
            assignment[beacon_id] = is_beacon_present_in_sample;
        }
        const BeaconPotential conditioned = beacon_potential.conditioned_on(assignment);

        // Plan assuming the current world
        const auto maybe_plan =
            compute_belief_road_map_plan(road_map, ekf, conditioned, options.brm_options);
        if (maybe_plan.has_value()) {
            plans.push_back(maybe_plan->nodes);
        }
    }

    const auto uncertainty_size =
        make_uncertainty_size<RobotBelief>(options.brm_options.uncertainty_size_options);

    // Evaluate the generated plans on each of the samples worlds
    std::vector<double> expected_size(plans.size(), 0.0);
    for (const auto &sample : world_samples) {
        const std::vector<RobotBelief> beliefs = evaluate_paths_with_configuration(
            plans, ekf, road_map, options.brm_options.max_sensor_range_m, sample);

        for (int i = 0; i < static_cast<int>(plans.size()); i++) {
            expected_size.at(i) += uncertainty_size(beliefs.at(i)) / world_samples.size();
        }

        if (options.timeout.has_value() &&
            time::current_robot_time() - plan_start_time > options.timeout.value()) {
            break;
        }
    }

    const auto min_iter = std::min_element(expected_size.begin(), expected_size.end());
    const int min_idx = std::distance(expected_size.begin(), min_iter);

    return {{
        .nodes = plans.at(min_idx),
        .log_probability_mass_tracked = math::logsumexp(log_probs),
    }};
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
