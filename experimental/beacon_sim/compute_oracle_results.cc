
#include <chrono>
#include <csignal>
#include <filesystem>
#include <thread>

#include "cxxopts.hpp"
#include "grpc++/grpc++.h"
#include "experimental/beacon_sim/work_server.hh"

namespace fs = std::filesystem;

namespace robot::experimental::beacon_sim {
volatile std::sig_atomic_t stop_requested = 0;
void signal_handler(int) { stop_requested = 1; }

void run_server(const fs::path &database_path, const fs::path &results_dir,
                const fs::path &experiment_config_path, const int server_port) {
    grpc::ServerBuilder builder;
    WorkServer service(database_path, results_dir, experiment_config_path);
    const std::string server_address = "0.0.0.0:" + std::to_string(server_port);
    builder.AddListeningPort(server_address,
                             grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::shared_ptr<grpc::Server> server(builder.BuildAndStart());

    std::thread([server](){
        while (!stop_requested) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        server->Shutdown();
    }).detach();

    std::cout << "Server listening on " << server_address << std::endl;
    server->Wait();
}

std::thread launch_server(const fs::path &results_dir, const fs::path &experiment_config_path,
                          const fs::path &database_path, const int server_port) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    return std::thread(run_server, results_dir, experiment_config_path, database_path, server_port);
}
}  // namespace robot::experimental::beacon_sim

int main(const int argc, const char **argv) {
    // clang-format off
    const char DEFAULT_NUM_THREADS[] = "0";
    const char DEFAULT_ADDRESS[] = "localhost";
    const char DEFAULT_PORT[] = "9001";
    cxxopts::Options options("compute oracle results",
                             "Compute oracle plans, optionally using many machines");
    options.add_options()
        ("results_dir", "Path to results directory", cxxopts::value<std::string>())
        ("database_path", "SQLite database with results", cxxopts::value<std::string>())
        ("experiment_config_path", "Path to experiment configs", cxxopts::value<std::string>())
        ("num_threads", "number of threads to use", 
            cxxopts::value<int>()->default_value(DEFAULT_NUM_THREADS))
        ("server_address", "address of job scheduler",
            cxxopts::value<std::string>()->default_value(DEFAULT_ADDRESS))
        ("server_port", "port of job scheduler",
            cxxopts::value<int>()->default_value(DEFAULT_PORT))
        ("launch_server", "launch job scheduler if flag is present")
        ("help" , "Print Usage");
    // clang-format on

    auto args = options.parse(argc, argv);

    if (args.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    const auto check_required = [&](const std::string &opt) {
        if (args.count(opt) == 0) {
            std::cout << "Missing " << opt << " argument" << std::endl;
            std::cout << options.help() << std::endl;
            std::exit(1);
        }
    };
    check_required("results_dir");
    check_required("database_path");
    check_required("experiment_config_path");

    std::optional<std::thread> server_thread;
    if (args["launch_server"].as<bool>()) {
        server_thread = robot::experimental::beacon_sim::launch_server(
            fs::path(args["database_path"].as<std::string>()),
            fs::path(args["results_dir"].as<std::string>()),
            fs::path(args["experiment_config_path"].as<std::string>()),
            args["server_port"].as<int>());
    }

    if (server_thread.has_value()) {
        server_thread->join();
    }
}
