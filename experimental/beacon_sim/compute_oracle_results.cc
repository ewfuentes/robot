
#include <grpc/grpc.h>
#include <unistd.h>

#include <chrono>
#include <csignal>
#include <filesystem>
#include <thread>

#include "BS_thread_pool.hpp"
#include "common/check.hh"
#include "common/time/robot_time_to_proto.hh"
#include "common/proto/load_from_file.hh"
#include "cxxopts.hpp"
#include "experimental/beacon_sim/belief_road_map_planner.hh"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/experiment_results.pb.h"
#include "experimental/beacon_sim/mapped_landmarks_to_proto.hh"
#include "experimental/beacon_sim/robot_belief.hh"
#include "experimental/beacon_sim/work_server.hh"
#include "experimental/beacon_sim/world_map_config_to_proto.hh"
#include "planning/belief_road_map.hh"
#include "planning/road_map.hh"
#include "planning/road_map_to_proto.hh"
#include "grpc++/grpc++.h"

namespace fs = std::filesystem;

namespace robot::experimental::beacon_sim {
volatile std::sig_atomic_t stop_requested = 0;
void signal_handler(int) { stop_requested = 1; }

void run_server(const fs::path &database_path, const fs::path &results_dir,
                const fs::path &experiment_config_path, const int server_port) {
    grpc::ServerBuilder builder;
    WorkServer service(database_path, results_dir, experiment_config_path);
    const std::string server_address = "0.0.0.0:" + std::to_string(server_port);
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::shared_ptr<grpc::Server> server(builder.BuildAndStart());

    std::thread([server]() {
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

proto::WorkServer::Stub get_client(const std::string server_address, const int server_port) {
    // Create a grpc client
    const std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
        server_address + ":" + std::to_string(server_port), grpc::InsecureChannelCredentials());

    constexpr bool TRY_TO_CONNECT = true;
    int state;
    while ((state = channel->GetState(TRY_TO_CONNECT)) != GRPC_CHANNEL_READY) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        std::cout << "Waiting for channel connection" << state << std::endl;
    }

    std::cout << "channel state: " << channel->GetState(false) << std::endl;

    return proto::WorkServer::Stub(channel);
}

std::optional<proto::GetJobResponse> get_job(const std::string &worker_name,
                                             proto::WorkServer::Stub &client) {
    grpc::ClientContext context;
    proto::GetJobRequest job_request;
    job_request.mutable_worker()->set_name(worker_name);
    proto::GetJobResponse job_response;
    CHECK(client.get_job(&context, job_request, &job_response).ok());
    return job_response;
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

std::tuple<BeaconPotential, planning::RoadMap, EkfSlam> load_environment(
    const proto::ExperimentConfig &config, const std::filesystem::path &config_path) {
    // Load Map Config
    const auto maybe_map_config = robot::proto::load_from_file<proto::WorldMapConfig>(
        config_path / config.map_config_path());
    CHECK(maybe_map_config.has_value(), config_path / config.map_config_path());
    const WorldMap world_map(unpack_from(maybe_map_config.value()));

    // Load EKF State
    const auto maybe_mapped_landmarks = robot::proto::load_from_file<proto::MappedLandmarks>(
        config_path / config.ekf_state_path());
    CHECK(maybe_mapped_landmarks.has_value());
    const auto mapped_landmarks = unpack_from(maybe_mapped_landmarks.value());

    // Create a road map
    const auto maybe_road_map = robot::proto::load_from_file<planning::proto::RoadMap>(
        config_path / config.road_map_path());
    CHECK(maybe_road_map.has_value());
    const planning::RoadMap road_map = unpack_from(maybe_road_map.value());

    return {world_map.beacon_potential(), road_map, load_ekf_slam(mapped_landmarks)};
}

std::vector<proto::Plan> compute_plans(
    const fs::path &results_file_path, [[maybe_unused]] const fs::path &experiment_config_path,
    [[maybe_unused]] const int start_idx, [[maybe_unused]] const int end_idx, const int num_threads,
    [[maybe_unused]] const std::function<void(const int, const int)> &progress_function) {
    const auto maybe_results_file =
        robot::proto::load_from_file<proto::ExperimentResult>(results_file_path);
    CHECK(maybe_results_file.has_value());
    const auto &results_file = maybe_results_file.value();

    const auto &experiment_config = results_file.experiment_config();
    // Load the beacon potential, road map and ekf from the config file
    const auto &[beacon_potential, road_map_, ekf_] =
        load_environment(experiment_config, experiment_config_path);

    // Load the beacon potential
    // compute the configuration samples
    // Fan out over samples
    BS::thread_pool pool(num_threads);
    return {};
}

void compute_oracle_results(const std::string server_address, const int server_port,
                            const int num_threads) {
    char hostname_char[256];
    gethostname(hostname_char, sizeof(hostname_char));
    const std::string hostname(hostname_char);

    proto::WorkServer::Stub client(get_client(server_address, server_port));

    std::optional<proto::GetJobResponse> maybe_job_response;
    while ((maybe_job_response = get_job(hostname, client)).has_value()) {
        std::cout << maybe_job_response.value().DebugString() << std::endl;
        const proto::JobInputs &inputs = maybe_job_response.value().job_inputs();
        // Setup progress callback
        time::RobotTimestamp job_start_time = time::current_robot_time();
        const auto progress_callback = [&client, &hostname, &job_start_time,
                                        job_id = maybe_job_response->job_id()](
                                           const int plans_completed, const int total_plans) {
            if (plans_completed % 10 != 0) {
                return;
            }
            grpc::ClientContext client_context;
            proto::JobStatusUpdateRequest update_request;
            proto::JobStatusUpdateResponse update_response;
            update_request.set_job_id(job_id);
            proto::JobStatusUpdate &update = *update_request.mutable_update();
            update.mutable_worker()->set_name(hostname);
            update.set_progress(static_cast<double>(plans_completed) / total_plans);
            pack_into(job_start_time, update.mutable_start_time());
            pack_into(time::current_robot_time(), update.mutable_current_time());
            CHECK(client.update_job_status(&client_context, update_request, &update_response).ok());
        };

        // Compute the results
        const auto plans = compute_plans(
            fs::path(inputs.results_file()), fs::path(inputs.experiment_config_path()),
            inputs.start_idx(), inputs.end_idx(), num_threads, progress_callback);
        // Send completed results
    }
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
    check_required("experiment_config_path");

    // Launch the server if necessary
    std::optional<std::thread> server_thread;
    if (args["launch_server"].as<bool>()) {
        check_required("database_path");
        server_thread = robot::experimental::beacon_sim::launch_server(
            fs::path(args["database_path"].as<std::string>()),
            fs::path(args["results_dir"].as<std::string>()),
            fs::path(args["experiment_config_path"].as<std::string>()),
            args["server_port"].as<int>());
    }

    // Launch the worker
    robot::experimental::beacon_sim::compute_oracle_results(
        args["server_address"].as<std::string>(), args["server_port"].as<int>(),
        args["num_threads"].as<int>());

    if (server_thread.has_value()) {
        server_thread->join();
    }
}
