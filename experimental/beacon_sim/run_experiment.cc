
#include "cxxopts.hpp"

#include <filesystem>

#include "common/check.hh"
#include "common/proto/load_from_file.hh"
#include "experimental/beacon_sim/experiment_config.pb.h"

using robot::experimental::beacon_sim::proto::ExperimentConfig;

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

    const auto &config_file = maybe_config_file.value();
    std::cout << config_file.DebugString() << std::endl;

}
