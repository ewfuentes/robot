
#include <filesystem>
#include <random>

#include "common/check.hh"
#include "common/proto/load_from_file.hh"
#include "common/argument_wrapper.hh"
#include "cxxopts.hpp"
#include "experimental/beacon_sim/experiment_config.pb.h"
#include "experimental/beacon_sim/mapped_landmarks.pb.h"
#include "experimental/beacon_sim/world_map_config.pb.h"
#include "planning/road_map.pb.h"

#include "Eigen/Core"

using robot::experimental::beacon_sim::proto::ExperimentConfig;

namespace robot::experimental::beacon_sim {
namespace {

struct StartGoal {
    Eigen::Vector2d start;
    Eigen::Vector2d goal;
}

std::vector<StartGoal> sample_start_goal(InOut<std::mt19937> gen) {
}

}

void run_experiment(const ExperimentConfig &config, const std::filesystem::path &base_path) {
    std::cout << config.DebugString() << std::endl;

    // Load Map Config
    const auto maybe_map_config =
        robot::proto::load_from_file<proto::WorldMapConfig>(base_path / config.map_config_path());
    CHECK(maybe_map_config.has_value());

    // Load EKF State
    const auto maybe_mapped_landmarks =
        robot::proto::load_from_file<proto::MappedLandmarks>(base_path / config.ekf_state_path());
    CHECK(maybe_mapped_landmarks.has_value());

    // Create a road map
    const auto maybe_road_map =
        robot::proto::load_from_file<planning::proto::RoadMap>(base_path / config.road_map_path());
    CHECK(maybe_road_map.has_value());

    for (const auto &planner_config : config.planner_configs()) {
        // Run the planner
        (void)planner_config;
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
