

#include <algorithm>
#include <chrono>
#include <execution>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

#include "common/liegroups/se2.hh"
#include "common/liegroups/se2_to_proto.hh"
#include "common/time/robot_time.hh"
#include "cxxopts.hpp"
#include "experimental/beacon_sim/beacon_sim_state.hh"
#include "experimental/beacon_sim/belief_road_map_planner.hh"
#include "experimental/beacon_sim/correlated_beacons.hh"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/mapped_landmarks.hh"
#include "experimental/beacon_sim/robot.hh"
#include "experimental/beacon_sim/rollout_statistics.pb.h"
#include "experimental/beacon_sim/sim_config.hh"
#include "experimental/beacon_sim/tick_sim.hh"
#include "experimental/beacon_sim/world_map.hh"
#include "experimental/beacon_sim/world_map_config_to_proto.hh"
#include "planning/probabilistic_road_map.hh"
#include "planning/road_map_to_proto.hh"

namespace robot::experimental::beacon_sim {
namespace {

struct TrialsConfig {
    std::filesystem::path output_path;
    Eigen::Vector2d goal_position;
    liegroups::SE2 local_from_robot;
    double p_beacon;
    double p_no_beacons;
    bool store_entire_plan;
};

std::vector<bool> configuration_from_index(const int idx, const int num_beacons) {
    std::vector<bool> out(num_beacons);
    for (int i = 0; i < num_beacons; i++) {
        out.at(i) = idx & (1 << i);
    }
    return out;
}

WorldMapConfig create_world_map_config(const double max_x_m, const double max_y_m,
                                       const double beacon_spacing_m, const double p_beacon,
                                       const double p_no_beacons) {
    // Create a box of beacons
    constexpr int VERTICAL_ID_OFFSET = 50;

    std::vector<Beacon> beacons;
    for (double pos = 0.0; pos < std::max(max_x_m, max_y_m); pos += beacon_spacing_m) {
        const int beacon_idx = pos / beacon_spacing_m;
        // Add beacons in a counterclockwise order
        if (pos < max_x_m) {
            // bottom left to bottom right
            beacons.push_back({
                .id = 2 * beacon_idx,
                .pos_in_local = {pos, 0.0},
            });

            // Top right to top left
            beacons.push_back({
                .id = 2 * beacon_idx + 1,
                .pos_in_local = {max_x_m - pos, max_y_m},
            });
        }

        if (pos < max_y_m) {
            // bottom right to top right
            beacons.push_back({
                .id = 2 * beacon_idx + VERTICAL_ID_OFFSET,
                .pos_in_local = {max_x_m, pos},
            });

            // Top left to bottom left
            beacons.push_back({
                .id = 2 * beacon_idx + 1 + VERTICAL_ID_OFFSET,
                .pos_in_local = {0.0, max_y_m - pos},
            });
        }
    }

    std::vector<int> ids;
    for (const auto &beacon : beacons) {
        ids.push_back(beacon.id);
    }

    return {
        .fixed_beacons = {},
        .blinking_beacons = {},
        .correlated_beacons =
            {
                .beacons = beacons,
                .potential = create_correlated_beacons({
                    .p_beacon = p_beacon,
                    .p_no_beacons = p_no_beacons,
                    .members = ids,
                }),
                // All beacons present by default
                .configuration = std::vector<bool>(ids.size(), true),

            },
        .obstacles = {},
    };
}

std::vector<std::tuple<int, WorldMapConfig>> create_all_beacon_presences(
    const WorldMapConfig &input) {
    // There are 2**N possible configurations with N beacons.
    // One way to assign whether the `k`'th beacon in present in configuration `C` is to consider
    // the `k`th bit in the binary representation of `C`. If this bit is set, we choose to erase
    // this element.
    const uint64_t num_configurations = 1 << input.correlated_beacons.beacons.size();
    const int num_beacons = input.correlated_beacons.beacons.size();

    std::vector<std::tuple<int, WorldMapConfig>> configs;
    configs.reserve(num_configurations);
    for (uint64_t config_idx = 0; config_idx < num_configurations; config_idx++) {
        // Create a copy of the input config
        configs.push_back(std::make_tuple(config_idx, input));
        std::get<1>(configs.back()).correlated_beacons.configuration =
            configuration_from_index(config_idx, num_beacons);
    }
    return configs;
}

EkfSlam create_ekf(const WorldMapConfig &map_config, const liegroups::SE2 &local_from_robot) {
    const EkfSlamConfig ekf_config = {
        .max_num_beacons = static_cast<int>(map_config.correlated_beacons.beacons.size()),
        .initial_beacon_uncertainty_m = 10.0,
        .along_track_process_noise_m_per_rt_meter = 0.1,
        .cross_track_process_noise_m_per_rt_meter = 0.01,
        .pos_process_noise_m_per_rt_s = 0.01,
        .heading_process_noise_rad_per_rt_meter = 0.1,
        .heading_process_noise_rad_per_rt_s = 0.0001,
        .beacon_pos_process_noise_m_per_rt_s = 1e-6,
        .range_measurement_noise_m = 0.5,
        .bearing_measurement_noise_rad = 0.1,
        .on_map_load_position_uncertainty_m = 0.5,
        .on_map_load_heading_uncertainty_rad = std::numbers::pi / 8.0,
    };

    // Create a mapped landmarks from the map_config
    const MappedLandmarks landmarks = [&]() {
        constexpr double BEACON_UNCERTAINTY_M = 0.001;
        constexpr double BEACON_VARIANCE_SQ_M = BEACON_UNCERTAINTY_M * BEACON_UNCERTAINTY_M;
        std::vector<int> beacon_ids;
        std::vector<Eigen::Vector2d> beacon_in_local;
        for (const auto &beacon : map_config.correlated_beacons.beacons) {
            beacon_ids.push_back(beacon.id);
            beacon_in_local.push_back(beacon.pos_in_local);
        }
        const int num_beacons = beacon_ids.size();
        return MappedLandmarks{
            .beacon_ids = std::move(beacon_ids),
            .beacon_in_local = std::move(beacon_in_local),
            .cov_in_local =
                Eigen::MatrixXd::Identity(2 * num_beacons, 2 * num_beacons) * BEACON_VARIANCE_SQ_M,
        };
    }();

    EkfSlam ekf(ekf_config, time::RobotTimestamp());
    ekf.estimate().local_from_robot(local_from_robot);
    constexpr bool LOAD_OFF_DIAGONALS = true;
    ekf.load_map(landmarks, LOAD_OFF_DIAGONALS);
    return ekf;
}

proto::RolloutStatistics compute_statistics(const std::vector<proto::BeaconSimDebug> &debug_msgs,
                                            const int trial_idx, const bool store_entire_plan) {
    proto::RolloutStatistics out;
    out.mutable_final_step()->CopyFrom(debug_msgs.back());
    out.set_trial_idx(trial_idx);
    if (store_entire_plan) {
        for (const auto &debug_msg : debug_msgs) {
            out.add_entire_plan()->CopyFrom(debug_msg);
        }
    }
    return out;
}
}  // namespace

void run_trials(const TrialsConfig &config) {
    constexpr double MAX_X_M = 20.0;
    constexpr double MAX_Y_M = 20.0;
    constexpr double BEACON_SPACING_M = MAX_X_M / 3.0;
    constexpr double ROAD_MAP_OFFSET_M = 5.0;
    constexpr double MAX_SENSOR_RANGE_M = 5.0;
    constexpr double UNCERTAINTY_TOLERANCE = 0.01;

    struct Map {
        bool in_free_space(const Eigen::Vector2d &) const { return true; }
        bool in_free_space(const Eigen::Vector2d &, const Eigen::Vector2d &) const { return true; }
        planning::MapBounds map_bounds() const {
            return {.bottom_left = {-ROAD_MAP_OFFSET_M, -ROAD_MAP_OFFSET_M},
                    .top_right = {MAX_X_M + ROAD_MAP_OFFSET_M, MAX_Y_M + ROAD_MAP_OFFSET_M}};
        }
    };

    // Create a world map
    const auto base_map_config = create_world_map_config(MAX_X_M, MAX_Y_M, BEACON_SPACING_M,
                                                         config.p_beacon, config.p_no_beacons);

    // Create a road map
    const auto road_map = planning::create_road_map(
        Map{}, {.seed = 0, .num_valid_points = 50, .desired_node_degree = 4});

    // Create an ekf
    const auto ekf = create_ekf(base_map_config, config.local_from_robot);

    // Compute the plan
    std::cout << "Starting to plan: " << std::endl;
    const BeliefRoadMapOptions options = {
        .max_sensor_range_m = MAX_SENSOR_RANGE_M,
        .uncertainty_tolerance = UNCERTAINTY_TOLERANCE,
        .max_num_edge_transforms = 1000,
        .timeout = std::nullopt,
    };
    const auto plan = compute_belief_road_map_plan(
        road_map, ekf, WorldMap(base_map_config).beacon_potential(), options);

    // Cycle through all possible assignments to beacon presence
    time::RobotTimestamp start = time::current_robot_time();

    const SimConfig sim_config = {
        .log_path = {},
        .map_input_path = {},
        .map_output_path = {},
        .world_map_config = {},
        .road_map_config = {},
        .goal_in_map = {},
        .map_from_initial_robot = {},
        .dt = std::chrono::milliseconds(25),
        .planner_config =
            robot::experimental::beacon_sim::BeliefRoadMapPlannerConfig{
                .allow_brm_backtracking = true,
            },
        .load_off_diagonals = false,
        .autostep = false,
        .correlated_beacons_configuration = {},
    };
    std::cout << "Plan: " << std::endl;
    for (const auto &node : plan->beliefs) {
        std::cout << node.local_from_robot.translation().transpose() << std::endl;
    }

    std::cout << "End Plan" << std::endl;

    const auto inputs = create_all_beacon_presences(base_map_config);
    std::vector<proto::RolloutStatistics> results(inputs.size());
    std::for_each(
        std::execution::par, inputs.begin(), inputs.end(),
        [&sim_config, &local_from_robot = config.local_from_robot,
         &goal_position = config.goal_position, &plan, &ekf, &road_map, &results,
         &store_entire_plan = config.store_entire_plan](const auto &input) {
            const auto &[idx, masked_map_config] = input;
            if (idx % 1000 == 0) {
                std::cout << "idx: " << idx << std::endl;
            }
            BeaconSimState state = {
                .time_of_validity = time::RobotTimestamp(),
                .map = WorldMap(masked_map_config),
                .road_map = road_map,
                .robot = RobotState(local_from_robot),
                .ekf = ekf,
                .observations = {},
                .goal = {{.time_of_validity = time::RobotTimestamp(),
                          .goal_position = goal_position}},
                .plan = {{.time_of_validity = time::RobotTimestamp(), .brm_plan = plan.value()}},
                .gen = std::mt19937(0),
            };
            const RobotCommand robot_command = {
                .turn_rad = 0.0,
                .move_m = 0.0,
            };

            int sim_iter = 0;
            std::optional<liegroups::SE2> maybe_prev_pose;
            std::vector<proto::BeaconSimDebug> debug_msgs;
            while (true) {
                state.time_of_validity = time::RobotTimestamp() + sim_config.dt * sim_iter;
                debug_msgs.emplace_back(tick_sim(sim_config, robot_command, make_in_out(state)));
                if (maybe_prev_pose.has_value() &&
                    (maybe_prev_pose->log() - state.robot.local_from_robot().log()).norm() == 0.0) {
                    // Not moving any more! call it done
                    break;
                }
                maybe_prev_pose = state.robot.local_from_robot();
                sim_iter++;
            }

            results[idx] = compute_statistics(debug_msgs, idx, store_entire_plan);
        });
    time::RobotTimestamp::duration dt = time::current_robot_time() - start;
    std::cout << std::chrono::duration<double>(dt).count() << std::endl;

    // Record results
    std::filesystem::create_directories(config.output_path.parent_path());
    proto::AllStatistics out;
    pack_into(base_map_config, out.mutable_world_map_config());
    pack_into(road_map, out.mutable_road_map());
    for (const auto &node : plan->nodes) {
        out.add_plan(node);
    }
    out.mutable_goal()->set_x(config.goal_position.x());
    out.mutable_goal()->set_y(config.goal_position.y());
    pack_into(ekf.estimate().local_from_robot(), out.mutable_local_from_start());
    if (config.store_entire_plan) {
        const auto single_trace_path =
            std::filesystem::path(config.output_path).replace_extension();
        std::filesystem::create_directories(single_trace_path);
        for (auto &trial_statistics : results) {
            // Write out a copy of the result with just this trace
            proto::AllStatistics single_trace_proto;
            single_trace_proto.CopyFrom(out);
            single_trace_proto.add_statistics()->CopyFrom(trial_statistics);

            // Clear the trace
            trial_statistics.clear_entire_plan();

            std::ofstream file_out(single_trace_path /
                                   (std::to_string(trial_statistics.trial_idx()) + ".pb"));
            single_trace_proto.SerializeToOstream(&file_out);
        }
    }
    for (auto &&trial_statistics : results) {
        out.mutable_statistics()->Add(std::move(trial_statistics));
    }
    std::ofstream file_out(config.output_path);
    out.SerializeToOstream(&file_out);
}
}  // namespace robot::experimental::beacon_sim

int main(const int argc, const char **argv) {
    const std::string DEFAULT_OUTPUT_PATH = "/tmp/results.pb";
    cxxopts::Options options("run_trials", "Run beacon sim trials");
    // clang-format off
    options.add_options()
      ("output", "Output path containing statistics for all rollouts",
         cxxopts::value<std::string>()->default_value(DEFAULT_OUTPUT_PATH))
      ("goal", "2D goal position (e.g. 3.0,4.5)", cxxopts::value<std::vector<double>>())
      ("local_from_start", "Starting robot pose X,Y,Theta", cxxopts::value<std::vector<double>>())
      ("p_beacon", "Marginal probability of a single beacon being present", cxxopts::value<double>())
      ("p_no_beacons", "Probability of no beacons being present", cxxopts::value<double>())
      ("store_entire_plan", "Save entire plan to output")
      ("help", "Print usage");
    // clang-format on

    auto args = options.parse(argc, argv);

    if (args.count("help")) {
        std::cout << options.help() << std::endl;
        std::exit(0);
    }

    if (args.count("goal") == 0) {
        std::cout << "--goal argument is required but missing." << std::endl;
        std::exit(1);
    } else if (args["goal"].as<std::vector<double>>().size() != 2) {
        std::cout << "Goal must be a 2D vector." << std::endl;
        std::exit(1);
    }

    if (args.count("local_from_start") == 0) {
        std::cout << "--local_from_start argument is required but missing." << std::endl;
        std::exit(1);
    } else if (args["local_from_start"].as<std::vector<double>>().size() != 3) {
        std::cout << "local_from_start must be a 3D vector." << std::endl;
        std::exit(1);
    }

    if (args.count("p_beacon") == 0) {
        std::cout << "--p_beacon argument is required but missing." << std::endl;
        std::exit(1);
    }

    if (args.count("p_no_beacons") == 0) {
        std::cout << "--p_no_beacons argument is required but missing." << std::endl;
        std::exit(1);
    }

    const auto &goal_position = args["goal"].as<std::vector<double>>();
    const auto &local_from_start_XYT = args["local_from_start"].as<std::vector<double>>();
    robot::experimental::beacon_sim::run_trials({
        .output_path = args["output"].as<std::string>(),
        .goal_position = Eigen::Vector2d{goal_position[0], goal_position[1]},
        .local_from_robot = robot::liegroups::SE2(
            local_from_start_XYT[2], {local_from_start_XYT[0], local_from_start_XYT[1]}),
        .p_beacon = args["p_beacon"].as<double>(),
        .p_no_beacons = args["p_no_beacons"].as<double>(),
        .store_entire_plan = args["store_entire_plan"].as<bool>(),
    });
}
