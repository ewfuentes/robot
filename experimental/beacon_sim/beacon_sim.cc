
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <utility>

#include "Eigen/Cholesky"
#include "common/argument_wrapper.hh"
#include "common/proto/load_from_file.hh"
#include "common/time/robot_time.hh"
#include "common/time/sim_clock.hh"
#include "cxxopts.hpp"
#include "experimental/beacon_sim/beacon_sim_debug.pb.h"
#include "experimental/beacon_sim/beacon_sim_state.hh"
#include "experimental/beacon_sim/belief_road_map_planner.hh"
#include "experimental/beacon_sim/correlated_beacons.hh"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/extract_mapped_landmarks.hh"
#include "experimental/beacon_sim/generate_observations.hh"
#include "experimental/beacon_sim/mapped_landmarks_to_proto.hh"
#include "experimental/beacon_sim/robot.hh"
#include "experimental/beacon_sim/sim_config.hh"
#include "experimental/beacon_sim/sim_log.pb.h"
#include "experimental/beacon_sim/tick_sim.hh"
#include "experimental/beacon_sim/visualize_beacon_sim.hh"
#include "experimental/beacon_sim/world_map.hh"
#include "planning/probabilistic_road_map.hh"
#include "visualization/gl_window/gl_window.hh"

using namespace std::literals::chrono_literals;

namespace robot::experimental::beacon_sim {

struct SimCommand {
    RobotCommand robot_command;
    double zoom_mult;
    bool should_exit;
    bool should_step;
    bool should_print_cov;
    bool time_travel_backward;
    bool time_travel_forward;
};

struct KeyCommand {
    bool arrow_left;
    bool arrow_right;
    bool arrow_up;
    bool arrow_down;
    bool q;
    bool p;
    bool left_bracket;
    bool right_bracket;
    bool minus;
    bool equal;

    static KeyCommand make_reset() {
        return KeyCommand{
            .arrow_left = false,
            .arrow_right = false,
            .arrow_up = false,
            .arrow_down = false,
            .q = false,
            .p = false,
            .left_bracket = false,
            .right_bracket = false,
            .minus = false,
            .equal = false,
        };
    }
};

SimCommand get_command(
    const KeyCommand key_command,
    const std::unordered_map<int, visualization::gl_window::JoystickState> &joysticks) {
    constexpr double TURN_AMOUNT_RAD = std::numbers::pi / 4.0;
    constexpr double MOVE_AMOUNT_M = 0.1;
    constexpr double ZOOM_FACTOR_ADJUST = 1.1;
    // Handle Keyboard Commands
    SimCommand out = {
        .robot_command = {.turn_rad = 0, .move_m = 0},
        .zoom_mult = 1.0,
        .should_exit = false,
        .should_step = false,
        .should_print_cov = false,
        .time_travel_backward = false,
        .time_travel_forward = false,
    };
    if (key_command.q) {
        out.should_exit = true;
    } else if (key_command.arrow_left) {
        out.robot_command.turn_rad = TURN_AMOUNT_RAD;
        out.should_step = true;
    } else if (key_command.arrow_right) {
        out.robot_command.turn_rad = -TURN_AMOUNT_RAD;
        out.should_step = true;
    } else if (key_command.arrow_up) {
        out.robot_command.move_m = MOVE_AMOUNT_M;
        out.should_step = true;
    } else if (key_command.arrow_down) {
        out.robot_command.move_m = -MOVE_AMOUNT_M;
        out.should_step = true;
    } else if (key_command.p) {
        out.should_print_cov = true;
    } else if (key_command.left_bracket) {
        out.time_travel_backward = true;
    } else if (key_command.right_bracket) {
        out.time_travel_forward = true;
    } else if (key_command.minus) {
        out.zoom_mult *= ZOOM_FACTOR_ADJUST;
    } else if (key_command.equal) {
        out.zoom_mult /= ZOOM_FACTOR_ADJUST;
    }

    for (const auto &[id, state] : joysticks) {
        if (state.buttons[GLFW_GAMEPAD_BUTTON_CIRCLE] == GLFW_PRESS) {
            out.should_exit = true;
            out.should_step = true;
        }
        const double left = -state.axes[GLFW_GAMEPAD_AXIS_LEFT_Y];
        const double right = -state.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y];

        const double mean = (left + right) / 2.0;
        const double mean_diff = (right - left) / 2.0;

        out.robot_command.turn_rad = 0.1 * mean_diff;
        out.robot_command.move_m = mean * 0.1, out.should_step = true;
    }
    return out;
}

// Creates a star centered on the origin.
std::vector<Eigen::Vector2d> create_star(const int num_points, const double aspect_ratio,
                                         const double point_dist) {
    std::vector<Eigen::Vector2d> out;
    const int num_steps = num_points * 2;
    for (int i = 0; i < num_steps; i++) {
        const double theta = (std::numbers::pi * i) / num_points;

        const double dist = point_dist * ((i % 2) == 0 ? 1.0 : aspect_ratio);

        out.emplace_back(Eigen::Vector2d{{dist * std::cos(theta), dist * std::sin(theta)}});
    }

    return out;
}

std::vector<Eigen::Vector2d> transform_points(std::vector<Eigen::Vector2d> &&pts_in_a,
                                              const liegroups::SE2 &b_from_a) {
    // Transform the elements in place
    std::transform(pts_in_a.begin(), pts_in_a.end(), pts_in_a.begin(),
                   [&b_from_a](const Eigen::Vector2d &pt_in_a) { return b_from_a * pt_in_a; });
    return pts_in_a;
}

WorldMapConfig world_map_config(const std::optional<int> &configuration) {
    WorldMapConfig out;
    // Add Beacons around the perimeter
    int beacon_id = 0;
    constexpr double TOP_EDGE_LENGTH_M = 28.0;
    constexpr double SIDE_EDGE_LENGTH_M = 20.0;
    constexpr int NUM_STEPS = 2;
    for (int i = 0; i < NUM_STEPS; i++) {
        const double alpha = static_cast<double>(i) / NUM_STEPS;
        // Top Edge from 0 to +x +y
        out.correlated_beacons.beacons.push_back(Beacon{
            .id = beacon_id++,
            .pos_in_local = {alpha * TOP_EDGE_LENGTH_M / 2.0, SIDE_EDGE_LENGTH_M / 2.0},
        });

        // Bottom Edge from +x -y to -x -y
        out.correlated_beacons.beacons.push_back(Beacon{
            .id = beacon_id++,
            .pos_in_local = {TOP_EDGE_LENGTH_M / 2.0 - alpha * TOP_EDGE_LENGTH_M,
                             -SIDE_EDGE_LENGTH_M / 2.0},
        });

        // Right Edge from +x +y to +x -y
        out.correlated_beacons.beacons.push_back(Beacon{
            .id = beacon_id++,
            .pos_in_local = {TOP_EDGE_LENGTH_M / 2.0, (1 - 2 * alpha) * SIDE_EDGE_LENGTH_M / 2.0},
        });
    }
    out.correlated_beacons.potential =
        create_correlated_beacons({.p_beacon = 0.5, .p_no_beacons = 0.25, .members = [&out]() {
                                       std::vector<int> ids;
                                       for (const auto &beacon : out.correlated_beacons.beacons) {
                                           ids.push_back(beacon.id);
                                       }
                                       return ids;
                                   }()});
    std::vector<bool> bool_config(out.correlated_beacons.beacons.size());
    for (int i = 0; i < static_cast<int>(bool_config.size()); i++) {
        if (configuration) {
            bool_config.at(i) = (configuration.value() & (1 << i));
        } else {
            bool_config.at(i) = true;
        }
    }
    out.correlated_beacons.configuration = bool_config;

    // Create obstacles
    {
        constexpr int NUM_STAR_POINTS = 5;
        constexpr double ASPECT_RATIO = 0.5;
        constexpr double POINT_LENGTH_M = 3.0;
        const liegroups::SE2 local_from_center(std::numbers::pi / 2.0, {5.0, 6.0});
        out.obstacles.obstacles.emplace_back(Obstacle(transform_points(
            create_star(NUM_STAR_POINTS, ASPECT_RATIO, POINT_LENGTH_M), local_from_center)));
    }
    {
        constexpr int NUM_STAR_POINTS = 3;
        constexpr double ASPECT_RATIO = 0.5;
        constexpr double POINT_LENGTH_M = 3.0;
        const liegroups::SE2 local_from_center(std::numbers::pi / 2.0, {-5.0, 6.0});
        out.obstacles.obstacles.emplace_back(Obstacle(transform_points(
            create_star(NUM_STAR_POINTS, ASPECT_RATIO, POINT_LENGTH_M), local_from_center)));
    }

    {
        constexpr int NUM_STAR_POINTS = 4;
        constexpr double ASPECT_RATIO = 0.5;
        constexpr double POINT_LENGTH_M = 3.0;
        const liegroups::SE2 local_from_center(0.0, {-5.0, -6.0});
        out.obstacles.obstacles.emplace_back(Obstacle(transform_points(
            create_star(NUM_STAR_POINTS, ASPECT_RATIO, POINT_LENGTH_M), local_from_center)));
    }

    {
        constexpr int NUM_STAR_POINTS = 6;
        constexpr double ASPECT_RATIO = 0.5;
        constexpr double POINT_LENGTH_M = 4.0;
        const liegroups::SE2 local_from_center(0.0, {5.0, -6.0});
        out.obstacles.obstacles.emplace_back(Obstacle(transform_points(
            create_star(NUM_STAR_POINTS, ASPECT_RATIO, POINT_LENGTH_M), local_from_center)));
    }

    return out;
}

void display_state(const BeaconSimState &state, const double zoom_factor,
                   InOut<visualization::gl_window::GlWindow> window) {
    const auto [screen_width_px, screen_height_px] = window->get_window_dims();
    const double aspect_ratio = static_cast<double>(screen_height_px) / screen_width_px;

    window->register_render_callback(
        [=]() { visualize_beacon_sim(state, zoom_factor, aspect_ratio); });
}

planning::RoadMap create_road_map(const WorldMap &map) {
    const planning::RoadmapCreationConfig config = {
        .seed = 0,
        .num_valid_points = 60,
        .desired_node_degree = 5,
    };
    struct MapInterface {
        const WorldMap *map_ptr;

        bool in_free_space(const Eigen::Vector2d &pt) const {
            for (const auto &obstacle : map_ptr->obstacles()) {
                if (obstacle.is_inside(pt)) {
                    return false;
                }
            }
            return true;
        }

        bool in_free_space(const Eigen::Vector2d &pt_a, const Eigen::Vector2d &pt_b) const {
            const Eigen::Vector2d delta = pt_b - pt_a;
            for (double alpha = 0; alpha <= 1; alpha += 0.01) {
                const Eigen::Vector2d query_pt = pt_a + alpha * delta;
                for (const auto &obstacle : map_ptr->obstacles()) {
                    if (obstacle.is_inside(query_pt)) {
                        return false;
                    }
                }
            }
            return true;
        }

        planning::MapBounds map_bounds() const {
            return planning::MapBounds{
                .bottom_left = Eigen::Vector2d{-15.0, -11.0},
                .top_right = Eigen::Vector2d{15.0, 11.0},
            };
        }
    };

    return create_road_map(MapInterface{.map_ptr = &map}, config,
                           {{-14.5, -10.5}, {14.5, -10.5}, {14.5, 10.5}, {-14.5, 10.5}});
}

void write_out_log_file(const SimConfig &sim_config,
                        std::vector<proto::BeaconSimDebug> debug_msgs) {
    proto::SimLog log_proto;
    std::cout << "Saving " << debug_msgs.size() << " debug messages to "
              << sim_config.log_path.value() << std::endl;
    for (proto::BeaconSimDebug &msg : debug_msgs) {
        log_proto.mutable_debug_msgs()->Add(std::move(msg));
    }
    std::fstream file_out(sim_config.log_path.value(),
                          std::ios::binary | std::ios::out | std::ios::trunc);
    log_proto.SerializeToOstream(&file_out);
}

void write_out_map(const SimConfig &sim_config, const EkfSlamEstimate &est) {
    proto::MappedLandmarks map_proto;
    std::cout << "Saving map to " << sim_config.map_output_path.value() << std::endl;
    pack_into(extract_mapped_landmarks(est), &map_proto);
    std::fstream file_out(sim_config.map_output_path.value(),
                          std::ios::binary | std::ios::out | std::ios::trunc);
    map_proto.SerializeToOstream(&file_out);
}

void load_map(const SimConfig &sim_config, InOut<EkfSlam> ekf_slam) {
    const std::filesystem::path map_path = sim_config.map_input_path.value();
    if (!std::filesystem::exists(map_path)) {
        std::cerr << "Tried load map from: " << map_path << " but it does not exist." << std::endl;
        return;
    }

    const auto maybe_proto = robot::proto::load_from_file<proto::MappedLandmarks>(map_path);
    if (!maybe_proto.has_value()) {
        std::cerr << "Unable to read file at: " << map_path << std::endl;
        return;
    }

    MappedLandmarks map = unpack_from(maybe_proto.value());
    ekf_slam->load_map(map, sim_config.load_off_diagonals);
}

void run_simulation(const SimConfig &sim_config) {
    bool run = true;
    time::set_default_time_provider(time::TimeProvider::SIM);

    visualization::gl_window::GlWindow gl_window(1920, 1440);

    // Initial robot state
    constexpr double INIT_POS_X_M = 0.0;
    constexpr double INIT_POS_Y_M = 0.0;
    constexpr double INIT_HEADING_RAD = 0.0;

    constexpr EkfSlamConfig EKF_CONFIG = {
        .max_num_beacons = 50,
        .initial_beacon_uncertainty_m = 100,
        .along_track_process_noise_m_per_rt_meter = 5e-2,
        .cross_track_process_noise_m_per_rt_meter = 1e-9,
        .pos_process_noise_m_per_rt_s = 1e-3,
        .heading_process_noise_rad_per_rt_meter = 1e-3,
        .heading_process_noise_rad_per_rt_s = 1e-10,
        .beacon_pos_process_noise_m_per_rt_s = 1e-3,
        .range_measurement_noise_m = 0.1,
        .bearing_measurement_noise_rad = 0.01,
        .on_map_load_position_uncertainty_m = 5.0,
        .on_map_load_heading_uncertainty_rad = 1.0,
    };

    WorldMap map = WorldMap(world_map_config(sim_config.correlated_beacons_configuration));
    BeaconSimState state = {
        .time_of_validity = time::current_robot_time(),
        .map = map,
        .road_map = create_road_map(map),
        .robot = RobotState(INIT_POS_X_M, INIT_POS_Y_M, INIT_HEADING_RAD),
        .ekf = EkfSlam(EKF_CONFIG, time::current_robot_time()),
        .observations = {},
        .goal = {{
            .time_of_validity = time::current_robot_time(),
            .goal_position = {-14.0, 0.0},
        }},
        .plan = std::nullopt,
        .gen = std::mt19937(0),
    };

    if (sim_config.map_input_path) {
        load_map(sim_config, make_in_out(state.ekf));
    }

    gl_window.register_window_resize_callback(
        [](const int width, const int height) { glViewport(0, 0, width, height); });

    std::atomic<KeyCommand> key_command{KeyCommand::make_reset()};
    gl_window.register_keyboard_callback(
        [&key_command](const int key, const int, const int action, const int) mutable {
            if (action == GLFW_RELEASE) {
                return;
            }
            auto update = key_command.load();
            if (key == GLFW_KEY_LEFT) {
                update.arrow_left = true;
            } else if (key == GLFW_KEY_RIGHT) {
                update.arrow_right = true;
            } else if (key == GLFW_KEY_UP) {
                update.arrow_up = true;
            } else if (key == GLFW_KEY_DOWN) {
                update.arrow_down = true;
            } else if (key == GLFW_KEY_Q) {
                update.q = true;
            } else if (key == GLFW_KEY_P) {
                update.p = true;
            } else if (key == GLFW_KEY_LEFT_BRACKET) {
                update.left_bracket = true;
            } else if (key == GLFW_KEY_RIGHT_BRACKET) {
                update.right_bracket = true;
            } else if (key == GLFW_KEY_EQUAL) {
                update.equal = true;
            } else if (key == GLFW_KEY_MINUS) {
                update.minus = true;
            }

            key_command.store(update);
        });

    std::vector<proto::BeaconSimDebug> debug_msgs;
    debug_msgs.reserve(10000);

    constexpr int MAX_STATE_QUEUE_SIZE = 500;
    std::deque<BeaconSimState> state_queue;
    state_queue.push_back(state);
    auto to_display_iter = state_queue.rbegin();
    double zoom_factor = 1.0;
    while (run) {
        // get command
        const auto command = get_command(key_command.exchange(KeyCommand::make_reset()),
                                         gl_window.get_joystick_states());

        if (command.should_exit) {
            run = false;
        }
        // Handle zoom
        zoom_factor *= command.zoom_mult;

        // Handle Covariance printing
        if (command.should_print_cov) {
            // Print out the covariance for each beacon
            std::cout << "************************************************ " << std::endl;
            const auto &ekf_estimate = state.ekf.estimate();
            for (const int beacon_id : ekf_estimate.beacon_ids) {
                std::cout << "================== Beacon: " << beacon_id << std::endl;
                std::cout << ekf_estimate.beacon_cov(beacon_id).value() << std::endl;
            }
        }

        // Handle Time Travel
        const int distance_to_newest = std::distance(state_queue.rbegin(), to_display_iter);
        const int distance_to_oldest = std::distance(to_display_iter, state_queue.rend());
        if (command.time_travel_backward || command.time_travel_forward) {
            std::cout << "Time travelling! " << distance_to_oldest << " / " << state_queue.size()
                      << std::endl;
        }
        if (command.time_travel_backward) {
            if (distance_to_oldest > 1) {
                to_display_iter++;
            } else {
                std::cout << "Can't go further back in time" << std::endl;
            }
        } else if (command.time_travel_forward) {
            if (distance_to_newest > 0) {
                to_display_iter--;
            } else {
                std::cout << "Already at present" << std::endl;
            }
        } else if (command.should_step && to_display_iter != state_queue.rbegin()) {
            state = *to_display_iter;
            std::cout << "Stepping Forward while in past! Starting new timeline and dropping "
                      << distance_to_newest << " future states." << std::endl;
            while (to_display_iter != state_queue.rbegin()) {
                state_queue.pop_back();
            }
            time::SimClock::reset();
            time::SimClock::advance(state.time_of_validity.time_since_epoch());
        }

        if (command.should_step || sim_config.autostep) {
            time::SimClock::advance(sim_config.dt);
            state.time_of_validity = time::current_robot_time();
            auto debug_msg = tick_sim(sim_config, command.robot_command, make_in_out(state));
            debug_msgs.emplace_back(std::move(debug_msg));

            state_queue.push_back(state);
            to_display_iter = state_queue.rbegin();
            while (static_cast<int>(state_queue.size()) > MAX_STATE_QUEUE_SIZE) {
                state_queue.pop_front();
            }
        }

        if (to_display_iter != state_queue.rend()) {
            display_state(*to_display_iter, zoom_factor, make_in_out(gl_window));
        }

        std::this_thread::sleep_for(sim_config.dt);
    }

    if (sim_config.log_path) {
        write_out_log_file(sim_config, std::move(debug_msgs));
    }
    if (sim_config.map_output_path) {
        write_out_map(sim_config, state.ekf.estimate());
    }
}
}  // namespace robot::experimental::beacon_sim

int main(int argc, char **argv) {
    const std::string DEFAULT_LOG_LOCATION = "/tmp/beacon_sim.pb";
    const std::string DEFAULT_MAP_SAVE_LOCATION = "/tmp/beacon_sim_map.pb";
    const std::string DEFAULT_MAP_LOAD_LOCATION = "/tmp/beacon_sim_map.pb";
    // clang-format off
    cxxopts::Options options("beacon_sim", "Simple Localization Simulator with Beacons");
    options.add_options()
      ("log_file", "Path to output file.", cxxopts::value<std::string>()->default_value(DEFAULT_LOG_LOCATION))
      ("map_output_path", "Path to save map file to" , cxxopts::value<std::string>()->default_value(DEFAULT_MAP_SAVE_LOCATION))
      ("map_input_path", "Path to load map file from", cxxopts::value<std::string>()->default_value(DEFAULT_MAP_LOAD_LOCATION))
      ("load_off_diagonals", "Whether off diagonal terms should be loaded from map")
      ("enable_brm_planner", "Generate BRM plan after each step")
      ("enable_info_lower_bound_planner", "Use the information lower bound planner")
      ("autostep", "automatically step the sim")
      ("allow_brm_backtracking", "Allow backtracking in BRM")
      ("correlated_beacons_config", "Desired Beacon Configuration. The ith beacon is present if the ith bit is set",
           cxxopts::value<int>())
      ("help", "Print usage");
    // clang-format on

    auto args = options.parse(argc, argv);
    if (args.count("help")) {
        std::cout << options.help() << std::endl;
        std::exit(0);
    }
    robot::experimental::beacon_sim::run_simulation({
        .log_path = args.count("log_file") ? std::make_optional(args["log_file"].as<std::string>())
                                           : std::nullopt,
        .map_input_path = args.count("map_input_path")
                              ? std::make_optional(args["map_input_path"].as<std::string>())
                              : std::nullopt,
        .map_output_path = args.count("map_output_path")
                               ? std::make_optional(args["map_output_path"].as<std::string>())
                               : std::nullopt,
        .dt = 25ms,
        .load_off_diagonals = args["load_off_diagonals"].as<bool>(),
        .enable_brm_planner = args["enable_brm_planner"].as<bool>(),
        .enable_info_lower_bound_planner = args["enable_info_lower_bound_planner"].as<bool>(),
        .allow_brm_backtracking = args["allow_brm_backtracking"].as<bool>(),
        .autostep = args["autostep"].as<bool>(),
        .correlated_beacons_configuration =
            args.count("correlated_beacons_config")
                ? std::make_optional(args["correlated_beacons_config"].as<int>())
                : std::nullopt,
    });
}
