
#include <algorithm>
#include <execution>
#include <filesystem>
#include <limits>
#include <optional>
#include <random>

#include "Eigen/Core"
#include "common/argument_wrapper.hh"
#include "common/check.hh"
#include "common/proto/load_from_file.hh"
#include "common/time/robot_time.hh"
#include "cxxopts.hpp"
#include "experimental/beacon_sim/beacon_potential.hh"
#include "experimental/beacon_sim/belief_road_map_planner.hh"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/experiment_config.pb.h"
#include "experimental/beacon_sim/make_belief_updater.hh"
#include "experimental/beacon_sim/mapped_landmarks_to_proto.hh"
#include "experimental/beacon_sim/robot_belief.hh"
#include "experimental/beacon_sim/world_map_config_to_proto.hh"
#include "planning/road_map_to_proto.hh"

using robot::experimental::beacon_sim::proto::ExperimentConfig;

namespace robot::experimental::beacon_sim {
namespace {

struct StartGoal {
    Eigen::Vector2d start;
    Eigen::Vector2d goal;
};

struct PlannerResult {
    struct Plan {
        std::vector<int> nodes;
        double log_prob_mass_tracked;
    };
    time::RobotTimestamp::duration elapsed_time;
    std::optional<Plan> plan;
};

std::vector<std::tuple<int, StartGoal>> sample_start_goal(const planning::RoadMap &map,
                                                          const int num_trials,
                                                          InOut<std::mt19937> gen) {
    Eigen::Vector2d min = {
        std::min_element(
            map.points().begin(), map.points().end(),
            [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) { return a.x() < b.x(); })
            ->x(),
        std::min_element(
            map.points().begin(), map.points().end(),
            [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) { return a.y() < b.y(); })
            ->y(),
    };
    Eigen::Vector2d max = {
        std::max_element(
            map.points().begin(), map.points().end(),
            [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) { return a.x() < b.x(); })
            ->x(),
        std::max_element(
            map.points().begin(), map.points().end(),
            [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) { return a.y() < b.y(); })
            ->y(),
    };

    std::vector<std::tuple<int, StartGoal>> out;
    std::uniform_real_distribution<> x_sampler(min.x(), max.x());
    std::uniform_real_distribution<> y_sampler(min.y(), max.y());
    for (int i = 0; i < num_trials; i++) {
        out.push_back(std::make_tuple(i, StartGoal{.start = {x_sampler(*gen), y_sampler(*gen)},
                                                   .goal = {x_sampler(*gen), y_sampler(*gen)}}));
    }
    return out;
}

EkfSlam load_ekf_slam(const MappedLandmarks &mapped_landmarks) {
    EkfSlam ekf(
        {
            .max_num_beacons = static_cast<int>(mapped_landmarks.beacon_ids.size()),
            .initial_beacon_uncertainty_m = 100,
            .along_track_process_noise_m_per_rt_meter = 5e-2,
            .cross_track_process_noise_m_per_rt_meter = 1e-9,
            .pos_process_noise_m_per_rt_s = 1e-3,
            .heading_process_noise_rad_per_rt_meter = 1e-3,
            .heading_process_noise_rad_per_rt_s = 1e-10,
            .beacon_pos_process_noise_m_per_rt_s = 1e-3,
            .range_measurement_noise_m = 0.1,
            .bearing_measurement_noise_rad = 0.01,
            .on_map_load_position_uncertainty_m = 1.0,
            .on_map_load_heading_uncertainty_rad = 1.0,
        },
        {});

    constexpr bool LOAD_OFF_DIAGONALS = true;
    ekf.load_map(mapped_landmarks, LOAD_OFF_DIAGONALS);

    return ekf;
}

PlannerResult run_planner(const planning::RoadMap &road_map, const EkfSlam &ekf,
                          const BeaconPotential &beacon_potential,
                          const proto::LandmarkBRMPlanner &config, const double max_sensor_range_m,
                          const std::optional<time::RobotTimestamp::duration> &timeout) {
    const time::RobotTimestamp start_time = time::current_robot_time();
    const auto plan = compute_landmark_belief_road_map_plan(
        road_map, ekf, beacon_potential,
        {.max_sensor_range_m = max_sensor_range_m,
         .sampled_belief_options =
             config.has_max_num_components()
                 ? std::make_optional(LandmarkBeliefRoadMapOptions::SampledBeliefOptions{
                       .max_num_components = config.max_num_components(), .seed = 12345})
                 : std::nullopt,
         .timeout = timeout});
    const time::RobotTimestamp::duration elapsed_time = time::current_robot_time() - start_time;

    // std::cout << "log probability mass tracked: "
    //           << plan->beliefs.back().log_probability_mass_tracked << std::endl;

    return {.elapsed_time = elapsed_time,
            .plan = plan.has_value() ? std::make_optional(PlannerResult::Plan{
                                           .nodes = plan->nodes,
                                           .log_prob_mass_tracked =
                                               plan->beliefs.back().log_probability_mass_tracked,
                                       })
                                     : std::nullopt};
}

PlannerResult run_planner(
    [[maybe_unused]] const planning::RoadMap &road_map, [[maybe_unused]] const EkfSlam &ekf,
    [[maybe_unused]] const BeaconPotential &beacon_potential,
    [[maybe_unused]] const proto::ExpectedBRMPlanner &config,
    [[maybe_unused]] const double max_sensor_range_m,
    [[maybe_unused]] const std::optional<time::RobotTimestamp::duration> &timeout) {
    // const auto plan = compute_landmark_belief_road_map_plan(
    //     road_map, ekf, beacon_potential,
    //     {.max_sensor_range_m = max_sensor_range_m,
    //      .sampled_belief_options =
    //          config.has_max_num_components()
    //              ? std::make_optional(LandmarkBeliefRoadMapOptions::SampledBeliefOptions{
    //                    .max_num_components = config.max_num_components(), .seed = 12345})
    //              : std::nullopt,
    //      .timeout = timeout
    //              });
    // CHECK(plan.has_value());
    // return plan->nodes;
    return {};
}

PlannerResult run_planner(const planning::RoadMap &road_map, const EkfSlam &ekf,
                          [[maybe_unused]] const BeaconPotential &beacon_potential,
                          [[maybe_unused]] const proto::OptimisticBRMPlanner &config,
                          const double max_sensor_range_m,
                          const std::optional<time::RobotTimestamp::duration> &timeout) {
    const BeliefRoadMapOptions options =
        BeliefRoadMapOptions{.max_sensor_range_m = max_sensor_range_m,
                             .uncertainty_tolerance = std::nullopt,
                             .max_num_edge_transforms = std::numeric_limits<int>::max(),
                             .timeout = timeout};
    const time::RobotTimestamp start_time = time::current_robot_time();
    const auto maybe_plan = compute_belief_road_map_plan(road_map, ekf, {}, options);
    const time::RobotTimestamp::duration elapsed_time = time::current_robot_time() - start_time;

    const double log_prob = beacon_potential.log_prob(beacon_potential.members());

    return {.elapsed_time = elapsed_time,
            .plan = maybe_plan.has_value() ? std::make_optional(PlannerResult::Plan{
                                                 .nodes = maybe_plan->nodes,
                                                 .log_prob_mass_tracked = log_prob,
                                             })
                                           : std::nullopt};
}

struct ComparisonResult {
    struct PlanStatistics {
        time::RobotTimestamp::duration elapsed_time;
        double expected_covariance_determinant;
        int num_successful_plans;
    };
    std::unordered_map<std::string, PlanStatistics> results;
    StartGoal start_goal;
};

ComparisonResult evaluate_results([[maybe_unused]] const StartGoal &start_goal,
                                  const WorldMap &world_map, const planning::RoadMap &road_map,
                                  const EkfSlam &ekf,
                                  const std::unordered_map<std::string, PlannerResult> &results,
                                  const int evaluation_seed, const int num_eval_trials,
                                  const double max_sensor_range_m) {
    std::mt19937 gen(evaluation_seed);

    std::unordered_map<std::string, ComparisonResult::PlanStatistics> results_map;
    for (int i = 0; i < num_eval_trials; i++) {
        const auto present_beacons = world_map.beacon_potential().sample(make_in_out(gen));
        const auto belief_updater = make_belief_updater(road_map, max_sensor_range_m, ekf,
                                                        present_beacons, TransformType::COVARIANCE);
        const RobotBelief initial_belief = {
            .local_from_robot = ekf.estimate().local_from_robot(),
            .cov_in_robot = ekf.estimate().robot_cov(),
        };
        for (const auto &[name, result] : results) {
            if (results_map.find(name) == results_map.end()) {
                results_map[name] = {
                    .elapsed_time = result.elapsed_time,
                    .expected_covariance_determinant = 0.0,
                    .num_successful_plans = 0,
                };
            }

            if (result.plan.has_value()) {
                RobotBelief belief = initial_belief;
                const auto &plan = result.plan->nodes;
                for (int i = 1; i < static_cast<int>(plan.size()); i++) {
                    belief = belief_updater(belief, plan.at(i - 1), plan.at(i));
                }

                results_map[name].num_successful_plans++;
                const double ratio = 1.0 / results_map[name].num_successful_plans;
                results_map[name].expected_covariance_determinant =
                    (1 - ratio) * results_map[name].expected_covariance_determinant +
                    ratio * belief.cov_in_robot.determinant();
            }
        }
    }

    return {
        .results = results_map,
        .start_goal = start_goal,
    };
}

}  // namespace

void run_experiment(const ExperimentConfig &config, const std::filesystem::path &base_path) {
    std::cout << config.DebugString() << std::endl;

    // Load Map Config
    const auto maybe_map_config =
        robot::proto::load_from_file<proto::WorldMapConfig>(base_path / config.map_config_path());
    CHECK(maybe_map_config.has_value());
    const WorldMap world_map(unpack_from(maybe_map_config.value()));

    // Load EKF State
    const auto maybe_mapped_landmarks =
        robot::proto::load_from_file<proto::MappedLandmarks>(base_path / config.ekf_state_path());
    CHECK(maybe_mapped_landmarks.has_value());
    const auto mapped_landmarks = unpack_from(maybe_mapped_landmarks.value());

    // Create a road map
    const auto maybe_road_map =
        robot::proto::load_from_file<planning::proto::RoadMap>(base_path / config.road_map_path());
    CHECK(maybe_road_map.has_value());
    planning::RoadMap road_map = unpack_from(maybe_road_map.value());

    std::mt19937 gen(config.start_goal_seed());
    const std::vector<std::tuple<int, StartGoal>> start_goals =
        sample_start_goal(road_map, config.num_trials(), make_in_out(gen));

    EkfSlam ekf = load_ekf_slam(mapped_landmarks);

    std::atomic<int> remaining = start_goals.size();
    const std::optional<time::RobotTimestamp::duration> timeout =
        config.has_plan_timeout_s() ? std::make_optional(time::as_duration(config.plan_timeout_s()))
                                    : std::nullopt;

    std::for_each(
        std::execution::par, start_goals.begin(), start_goals.end(),
        [=, &remaining, eval_base_seed = config.evaluation_base_seed(),
         num_eval_trials = config.num_eval_trials(),
         max_sensor_range_m = config.max_sensor_range_m()](const auto &elem) mutable {
            const auto &[idx, start_goal] = elem;
            // Add the start goal to the road map
            road_map.add_start_goal(
                {.start = start_goal.start,
                 .goal = start_goal.goal,
                 .connection_radius_m = config.start_goal_connection_radius_m()});

            // Set the initial pose
            const liegroups::SE2 local_from_robot = liegroups::SE2::trans(start_goal.start);
            ekf.estimate().local_from_robot(local_from_robot);

            std::unordered_map<std::string, PlannerResult> results;

            for (const auto &planner_config : config.planner_configs()) {
                CHECK(planner_config.planner_config_oneof_case() !=
                      proto::PlannerConfig::PLANNER_CONFIG_ONEOF_NOT_SET);
                switch (planner_config.planner_config_oneof_case()) {
                    case proto::PlannerConfig::kLandmarkBrmConfig: {
                        results[planner_config.name()] = run_planner(
                            road_map, ekf, world_map.beacon_potential(),
                            planner_config.landmark_brm_config(), max_sensor_range_m, timeout);
                        break;
                    }
                    case proto::PlannerConfig::kOptimisticBrmConfig: {
                        results[planner_config.name()] = run_planner(
                            road_map, ekf, world_map.beacon_potential(),
                            planner_config.optimistic_brm_config(), max_sensor_range_m, timeout);
                        break;
                    }
                    case proto::PlannerConfig::kExpectedBrmConfig: {
                        results[planner_config.name()] = run_planner(
                            road_map, ekf, world_map.beacon_potential(),
                            planner_config.expected_brm_config(), max_sensor_range_m, timeout);
                        break;
                    }
                    default: {
                        CHECK(false, "Unhandled Planner Config type",
                              planner_config.planner_config_oneof_case());
                    }
                }
            }

            evaluate_results(start_goal, world_map, road_map, ekf, results, eval_base_seed + idx,
                             num_eval_trials, max_sensor_range_m);
            std::cout << "remaining: " << --remaining << std::endl;
        });

    // Write out the results
}
}  // namespace robot::experimental::beacon_sim

int main(int argc, const char **argv) {
    // clang-format off
    cxxopts::Options options("run_experiment", "Run experiments for paper");
    options.add_options()
        ("config_file", "Path to experiment config file", cxxopts::value<std::string>())
        ("help", "Print usage");
    // clang-format on

    auto args = options.parse(argc, argv);

    if (args.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if (args.count("config_file") == 0) {
        std::cout << "Missing config_file argument" << std::endl;
        std::cout << options.help() << std::endl;
        return 1;
    }

    std::filesystem::path config_file_path = args["config_file"].as<std::string>();
    const auto maybe_config_file = robot::proto::load_from_file<ExperimentConfig>(config_file_path);
    CHECK(maybe_config_file.has_value());

    robot::experimental::beacon_sim::run_experiment(maybe_config_file.value(),
                                                    config_file_path.remove_filename());
}
