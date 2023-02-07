

#include <algorithm>
#include <chrono>
#include <csignal>
#include <execution>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

#include "common/liegroups/se2.hh"
#include "common/time/robot_time.hh"
#include "cxxopts.hpp"
#include "experimental/beacon_sim/beacon_sim_state.hh"
#include "experimental/beacon_sim/belief_road_map_planner.hh"
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
bool debug = true;
namespace {

void sigusr_1_handler(int) { debug = true; }

WorldMapConfig create_world_map_config(const double max_x_m, const double max_y_m) {
    // Create a box of beacons
    constexpr double BEACON_SPACING_M = 5.0;
    constexpr int VERTICAL_ID_OFFSET = 50;

    std::vector<Beacon> beacons;
    for (double pos = 0.0; pos < std::max(max_x_m, max_y_m); pos += BEACON_SPACING_M) {
        const int beacon_idx = pos / BEACON_SPACING_M;
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

    return {.fixed_beacons = {beacons}, .blinking_beacons = {}, .obstacles = {}};
}

std::vector<std::tuple<int, WorldMapConfig>> create_all_beacon_presences(
    const WorldMapConfig &input) {
    // There are 2**N possible configurations with N beacons.
    // One way to assign whether the `k`'th beacon in present in configuration `C` is to consider
    // the `k`th bit in the binary representation of `C`. If this bit is set, we choose to erase
    // this element.
    const uint64_t num_configurations = 1 << input.fixed_beacons.beacons.size();
    const int num_beacons = input.fixed_beacons.beacons.size();

    std::vector<std::tuple<int, WorldMapConfig>> configs;
    configs.reserve(num_configurations);
    for (uint64_t config_idx = 0; config_idx < num_configurations; config_idx++) {
        // Create a copy of the input config
        configs.push_back(std::make_tuple(config_idx, input));
        auto &beacons = std::get<1>(configs.back()).fixed_beacons.beacons;
        for (int beacon_idx = num_beacons - 1; beacon_idx >= 0; beacon_idx--) {
            if (config_idx & (1 << beacon_idx)) {
                // If the `k`th bit is set, erase it
                beacons.erase(beacons.begin() + beacon_idx);
            }
        }
    }
    return configs;
}

EkfSlam create_ekf(const WorldMapConfig &map_config) {
    const EkfSlamConfig ekf_config = {
        .max_num_beacons = static_cast<int>(map_config.fixed_beacons.beacons.size()),
        .initial_beacon_uncertainty_m = 10.0,
        .along_track_process_noise_m_per_rt_meter = 0.1,
        .cross_track_process_noise_m_per_rt_meter = 0.01,
        .pos_process_noise_m_per_rt_s = 0.01,
        .heading_process_noise_rad_per_rt_meter = 0.001,
        .heading_process_noise_rad_per_rt_s = 0.0001,
        .beacon_pos_process_noise_m_per_rt_s = 1e-6,
        .range_measurement_noise_m = 0.1,
        .bearing_measurement_noise_rad = 0.001,
        .on_map_load_position_uncertainty_m = 0.5,
        .on_map_load_heading_uncertainty_rad = std::numbers::pi / 8.0,
    };

    // Create a mapped landmarks from the map_config
    const MappedLandmarks landmarks = [&]() {
        constexpr double BEACON_UNCERTAINTY_M = 0.001;
        constexpr double BEACON_VARIANCE_SQ_M = BEACON_UNCERTAINTY_M * BEACON_UNCERTAINTY_M;
        std::vector<int> beacon_ids;
        std::vector<Eigen::Vector2d> beacon_in_local;
        for (const auto &beacon : map_config.fixed_beacons.beacons) {
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
    constexpr bool LOAD_OFF_DIAGONALS = true;
    ekf.load_map(landmarks, LOAD_OFF_DIAGONALS);
    return ekf;
}

proto::RolloutStatistics compute_statistics(const std::vector<proto::BeaconSimDebug> &debug_msgs) {
    proto::RolloutStatistics out;
    out.mutable_final_step()->CopyFrom(debug_msgs.back());
    return out;
}

}  // namespace
void run_trials() {
    constexpr double MAX_X_M = 20.0;
    constexpr double MAX_Y_M = 20.0;
    constexpr double ROAD_MAP_OFFSET_M = 5.0;
    constexpr double MAX_SENSOR_RANGE_M = 5.0;
    constexpr int NUM_START_CONNECTIONS = 1;
    constexpr int NUM_GOAL_CONNECTIONS = 1;
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
    const auto base_map_config = create_world_map_config(MAX_X_M, MAX_Y_M);

    // Create a road map
    const auto road_map = planning::create_road_map(
        Map{}, {.seed = 0, .num_valid_points = 50, .desired_node_degree = 4});

    // Create an ekf
    const auto ekf = create_ekf(base_map_config);

    // Provide a goal
    const Eigen::Vector2d goal_state = {0.0, MAX_Y_M + ROAD_MAP_OFFSET_M};

    // Compute the plan
    std::cout << "Starting to plan: " << std::endl;
    const auto plan = compute_belief_road_map_plan(road_map, ekf, goal_state, MAX_SENSOR_RANGE_M,
                                                   NUM_START_CONNECTIONS, NUM_GOAL_CONNECTIONS,
                                                   UNCERTAINTY_TOLERANCE);
    std::cout << "Done planning" << std::endl;

    // Cycle through all possible assignments to beacon presence
    time::RobotTimestamp start = time::current_robot_time();

    const SimConfig sim_config = {
        .log_path = {},
        .map_input_path = {},
        .map_output_path = {},
        .dt = std::chrono::milliseconds(25),
        .load_off_diagonals = false,
        .enable_brm_planner = true,
        .autostep = false,
    };
    std::cout << "Plan: " << std::endl;
    for (const auto &node : plan->beliefs) {
        std::cout << node.local_from_robot.translation().transpose() << std::endl;
    }

    std::cout << "End Plan" << std::endl;

    const auto inputs = create_all_beacon_presences(base_map_config);
    std::filesystem::create_directories("/tmp/beacon_sim_log/");
    std::vector<proto::RolloutStatistics> results(inputs.size());
    std::for_each(
        std::execution::par, inputs.begin(), inputs.end(),
        [&sim_config, &goal_state, &plan, &ekf, &road_map, &results](const auto &input) {
            const auto &[idx, masked_map_config] = input;
            if (idx % 1000 == 0) {
                std::cout << "idx: " << idx << std::endl;
            }
            BeaconSimState state = {
                .time_of_validity = time::RobotTimestamp(),
                .map = WorldMap(masked_map_config),
                .road_map = road_map,
                .robot = RobotState(liegroups::SE2()),
                .ekf = ekf,
                .observations = {},
                .goal = {{.time_of_validity = time::RobotTimestamp(), .goal_state = goal_state}},
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

            results[idx] = compute_statistics(debug_msgs);
        });
    time::RobotTimestamp::duration dt = time::current_robot_time() - start;
    std::cout << std::chrono::duration<double>(dt).count() << std::endl;

    // Record results
    proto::AllStatistics out;
    pack_into(base_map_config, out.mutable_world_map_config());
    pack_into(road_map, out.mutable_road_map());
    for (auto &&trial_statistics : results) {
        out.mutable_statistics()->Add(std::move(trial_statistics));
    }
    std::ofstream file_out("/tmp/beacon_sim_log/results.pb");
    out.SerializeToOstream(&file_out);
}
}  // namespace robot::experimental::beacon_sim

int main() {
    std::signal(SIGUSR1, robot::experimental::beacon_sim::sigusr_1_handler);
    robot::experimental::beacon_sim::run_trials();
}
