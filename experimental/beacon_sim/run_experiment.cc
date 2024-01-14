
#include <algorithm>
#include <filesystem>
#include <limits>
#include <random>

#include "Eigen/Core"
#include "common/argument_wrapper.hh"
#include "common/check.hh"
#include "common/proto/load_from_file.hh"
#include "cxxopts.hpp"
#include "experimental/beacon_sim/beacon_potential.hh"
#include "experimental/beacon_sim/belief_road_map_planner.hh"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/experiment_config.pb.h"
#include "experimental/beacon_sim/mapped_landmarks_to_proto.hh"
#include "experimental/beacon_sim/world_map_config_to_proto.hh"
#include "planning/road_map_to_proto.hh"

using robot::experimental::beacon_sim::proto::ExperimentConfig;

namespace robot::experimental::beacon_sim {
namespace {

struct StartGoal {
    Eigen::Vector2d start;
    Eigen::Vector2d goal;
};

std::vector<StartGoal> sample_start_goal(const planning::RoadMap &map, const int num_trials,
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

    std::vector<StartGoal> out;
    std::uniform_real_distribution<> x_sampler(min.x(), max.x());
    std::uniform_real_distribution<> y_sampler(min.y(), max.y());
    for (int i = 0; i < num_trials; i++) {
        out.push_back({.start = {x_sampler(*gen), y_sampler(*gen)},
                       .goal = {x_sampler(*gen), y_sampler(*gen)}});
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

std::vector<int> run_planner(const planning::RoadMap &road_map, const EkfSlam &ekf,
                             const BeaconPotential &beacon_potential,
                             const proto::LandmarkBRMPlanner &config,
                             const double max_sensor_range_m) {
    if (config.has_max_num_components()) {
        std::cout << "landmark belief components max: " << config.max_num_components() << std::endl;
    } else {
        std::cout << "landmark belief with unlimited num components" << std::endl;
    }
    const auto plan = compute_landmark_belief_road_map_plan(
        road_map, ekf, beacon_potential,
        {.max_sensor_range_m = max_sensor_range_m,
         .sampled_belief_options =
             config.has_max_num_components()
                 ? std::make_optional(LandmarkBeliefRoadMapOptions::SampledBeliefOptions{
                       .max_num_components = config.max_num_components(), .seed = 12345})
                 : std::nullopt});
    CHECK(plan.has_value());

    std::cout << "log probability mass tracked: "
              << plan->beliefs.back().log_probability_mass_tracked << std::endl;
    return plan->nodes;
}

std::vector<int> run_planner([[maybe_unused]] const planning::RoadMap &road_map,
                             [[maybe_unused]] const EkfSlam &ekf,
                             [[maybe_unused]] const BeaconPotential &beacon_potential,
                             [[maybe_unused]] const proto::ExpectedBRMPlanner &config,
                             [[maybe_unused]] const double max_sensor_range_m) {
    // const auto plan = compute_landmark_belief_road_map_plan(
    //     road_map, ekf, beacon_potential,
    //     {.max_sensor_range_m = max_sensor_range_m,
    //      .sampled_belief_options =
    //          config.has_max_num_components()
    //              ? std::make_optional(LandmarkBeliefRoadMapOptions::SampledBeliefOptions{
    //                    .max_num_components = config.max_num_components(), .seed = 12345})
    //              : std::nullopt});
    // CHECK(plan.has_value());
    // return plan->nodes;
    return {};
}

std::vector<int> run_planner(const planning::RoadMap &road_map, const EkfSlam &ekf,
                             [[maybe_unused]] const BeaconPotential &beacon_potential,
                             [[maybe_unused]] const proto::OptimisticBRMPlanner &config,
                             const double max_sensor_range_m) {
    const BeliefRoadMapOptions options = BeliefRoadMapOptions{
        .max_sensor_range_m = max_sensor_range_m,
        .uncertainty_tolerance = std::nullopt,
        .max_num_edge_transforms = std::numeric_limits<int>::max(),
    };
    const auto maybe_plan = compute_belief_road_map_plan(road_map, ekf, {}, options);
    CHECK(maybe_plan.has_value());

    return maybe_plan->nodes;
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
    const std::vector<StartGoal> start_goals =
        sample_start_goal(road_map, config.num_trials(), make_in_out(gen));

    EkfSlam ekf = load_ekf_slam(mapped_landmarks);

    for (const auto &start_goal : start_goals) {
        // Add the start goal to the road map
        road_map.add_start_goal({.start = start_goal.start,
                                 .goal = start_goal.goal,
                                 .connection_radius_m = config.start_goal_connection_radius_m()});

        // Set the initial pose
        const liegroups::SE2 local_from_robot = liegroups::SE2::trans(start_goal.start);
        ekf.estimate().local_from_robot(local_from_robot);

        std::unordered_map<std::string, std::vector<int>> results;

        for (const auto &planner_config : config.planner_configs()) {
            CHECK(planner_config.planner_config_oneof_case() !=
                  proto::PlannerConfig::PLANNER_CONFIG_ONEOF_NOT_SET);
            switch (planner_config.planner_config_oneof_case()) {
                case proto::PlannerConfig::kLandmarkBrmConfig: {
                    results[planner_config.name()] = run_planner(
                        road_map, ekf, world_map.beacon_potential(),
                        planner_config.landmark_brm_config(), planner_config.max_sensor_range_m());
                    break;
                }
                case proto::PlannerConfig::kOptimisticBrmConfig: {
                    results[planner_config.name()] =
                        run_planner(road_map, ekf, world_map.beacon_potential(),
                                    planner_config.optimistic_brm_config(),
                                    planner_config.max_sensor_range_m());
                    break;
                }
                case proto::PlannerConfig::kExpectedBrmConfig: {
                    results[planner_config.name()] = run_planner(
                        road_map, ekf, world_map.beacon_potential(),
                        planner_config.expected_brm_config(), planner_config.max_sensor_range_m());
                    break;
                }
                default: {
                    CHECK(false, "Unhandled Planner Config type",
                          planner_config.planner_config_oneof_case());
                }
            }
        }
        // evaluate_results(world_map, road_map, results);
    }

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
