
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
#include "common/liegroups/se2_to_proto.hh"
#include "common/proto/load_from_file.hh"
#include "common/time/robot_time.hh"
#include "common/time/robot_time_to_proto.hh"
#include "common/time/sim_clock.hh"
#include "cxxopts.hpp"
#include "experimental/beacon_sim/beacon_observation_to_proto.hh"
#include "experimental/beacon_sim/beacon_sim_debug.pb.h"
#include "experimental/beacon_sim/beacon_sim_state.hh"
#include "experimental/beacon_sim/belief_road_map_planner.hh"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/ekf_slam_estimate_to_proto.hh"
#include "experimental/beacon_sim/extract_mapped_landmarks.hh"
#include "experimental/beacon_sim/generate_observations.hh"
#include "experimental/beacon_sim/mapped_landmarks_to_proto.hh"
#include "experimental/beacon_sim/robot.hh"
#include "experimental/beacon_sim/sim_log.pb.h"
#include "experimental/beacon_sim/visualize_beacon_sim.hh"
#include "experimental/beacon_sim/world_map.hh"
#include "planning/probabilistic_road_map.hh"
#include "visualization/gl_window/gl_window.hh"

using namespace std::literals::chrono_literals;

namespace robot::experimental::beacon_sim {
namespace {
const auto DT = 25ms;

struct SimConfig {
    std::optional<std::string> log_path;
    std::optional<std::string> map_input_path;
    std::optional<std::string> map_output_path;
    bool load_off_diagonals;
    bool enable_brm_planner;
    bool autostep;
};

}  // namespace

struct RobotCommand {
    double turn_rad;
    double move_m;
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

RobotCommand get_command(
    const KeyCommand key_command,
    const std::unordered_map<int, visualization::gl_window::JoystickState> &joysticks) {
    constexpr double TURN_AMOUNT_RAD = std::numbers::pi / 4.0;
    constexpr double MOVE_AMOUNT_M = 0.1;
    constexpr double ZOOM_FACTOR_ADJUST = 1.1;
    // Handle Keyboard Commands
    RobotCommand out = {
        .turn_rad = 0,
        .move_m = 0,
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
        out.turn_rad = TURN_AMOUNT_RAD;
        out.should_step = true;
    } else if (key_command.arrow_right) {
        out.turn_rad = -TURN_AMOUNT_RAD;
        out.should_step = true;
    } else if (key_command.arrow_up) {
        out.move_m = MOVE_AMOUNT_M;
        out.should_step = true;
    } else if (key_command.arrow_down) {
        out.move_m = -MOVE_AMOUNT_M;
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

        out.turn_rad = 0.1 * mean_diff;
        out.move_m = mean * 0.1, out.should_step = true;
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

WorldMapConfig world_map_config() {
    WorldMapConfig out;
    // Add Beacons around the perimeter
    int beacon_id = 0;
    constexpr double TOP_EDGE_LENGTH_M = 28.0;
    constexpr double SIDE_EDGE_LENGTH_M = 20.0;
    constexpr int NUM_STEPS = 2;
    for (int i = 0; i < NUM_STEPS; i++) {
        const double alpha = static_cast<double>(i) / NUM_STEPS;
        // Top Edge from 0 to +x +y
        out.fixed_beacons.beacons.push_back(Beacon{
            .id = beacon_id++,
            .pos_in_local = {alpha * TOP_EDGE_LENGTH_M / 2.0, SIDE_EDGE_LENGTH_M / 2.0},
        });

        // Bottom Edge from +x -y to -x -y
        out.fixed_beacons.beacons.push_back(Beacon{
            .id = beacon_id++,
            .pos_in_local = {TOP_EDGE_LENGTH_M / 2.0 - alpha * TOP_EDGE_LENGTH_M,
                             -SIDE_EDGE_LENGTH_M / 2.0},
        });

        // Right Edge from +x +y to +x -y
        out.fixed_beacons.beacons.push_back(Beacon{
            .id = beacon_id++,
            .pos_in_local = {TOP_EDGE_LENGTH_M / 2.0, (1 - 2 * alpha) * SIDE_EDGE_LENGTH_M / 2.0},
        });
    }

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

proto::BeaconSimDebug tick_sim(const RobotCommand &command, InOut<BeaconSimState> state,
                               const SimConfig &config) {
    constexpr ObservationConfig OBS_CONFIG = {
        .range_noise_std_m = 0.1,
        .max_sensor_range_m = 5.0,
    };

    const Eigen::Vector2d goal_state{-14.0, 0.0};

    RobotCommand next_command = command;
    if (config.enable_brm_planner) {
        // Plan
        if (!state->plan.has_value()) {
            constexpr int NUM_START_CONNECTIONS = 6;
            constexpr int NUM_GOAL_CONNECTIONS = 6;
            constexpr double UNCERTAINTY_TOLERANCE = 0.1;
            std::cout << "Starting to Plan" << std::endl;
            const auto brm_plan = compute_belief_road_map_plan(
                state->road_map, state->ekf, goal_state, OBS_CONFIG.max_sensor_range_m.value(),
                NUM_START_CONNECTIONS, NUM_GOAL_CONNECTIONS, UNCERTAINTY_TOLERANCE);
            std::cout << "plan complete" << std::endl;
            for (int idx = 0; idx < static_cast<int>(brm_plan->nodes.size()); idx++) {
                std::cout << idx << " " << brm_plan->nodes.at(idx) << " "
                          << brm_plan->beliefs.at(idx).local_from_robot.translation().transpose()
                          << " cov det: " << brm_plan->beliefs.at(idx).cov_in_robot.determinant()
                          << std::endl;
            }
            state->plan = {.time_of_validity = state->time_of_validity,
                           .brm_plan = brm_plan.value()};
        }
        if (state->plan.has_value()) {
            // Figure out which which node we should be targeting
            constexpr double MAX_SPEED_MPS = 5.0;
            const auto time_since_plan_tov =
                state->time_of_validity - state->plan->time_of_validity;
            const std::vector<RobotBelief> &beliefs = state->plan->brm_plan.beliefs;
            time::RobotTimestamp::duration time_along_plan = std::chrono::seconds(0);
            for (int i = 1; i < static_cast<int>(beliefs.size()); i++) {
                const liegroups::SE2 &local_from_prev_robot = beliefs.at(i - 1).local_from_robot;
                const liegroups::SE2 &local_from_next_robot = beliefs.at(i).local_from_robot;
                const liegroups::SE2 next_from_prev =
                    local_from_next_robot.inverse() * local_from_prev_robot;
                const double dist_m = next_from_prev.arclength();
                const time::RobotTimestamp::duration step_dt =
                    time::as_duration(dist_m / MAX_SPEED_MPS);
                if (time_along_plan + step_dt > time_since_plan_tov) {
                    const liegroups::SE2 local_from_est_robot =
                        state->ekf.estimate().local_from_robot();
                    const Eigen::Vector2d target_in_robot =
                        local_from_est_robot.inverse() * local_from_next_robot.translation();

                    std::cout << "Time since plan tov: " << time_since_plan_tov
                              << " next target tov: " << time_along_plan + step_dt << std::endl;

                    std::cout << "local_from_target: "
                              << local_from_next_robot.translation().transpose() << std::endl;
                    std::cout << "local_from_robot: "
                              << local_from_est_robot.translation().transpose() << std::endl;
                    std::cout << "target_from_robot: " << target_in_robot.transpose() << std::endl;

                    if (target_in_robot.norm() > 0.1)  {
                        next_command.turn_rad =
                            std::atan2(target_in_robot.y(), target_in_robot.x());
                    }
                    next_command.move_m =
                        std::min(MAX_SPEED_MPS * std::chrono::duration<double>(DT).count(),
                                 target_in_robot.norm());
                    std::cout << " Move amount: " << next_command.move_m;
                    std::cout << " turn amount: " << next_command.turn_rad << std::endl;
                    break;
                } else {
                    time_along_plan += step_dt;
                }
            }
        }
    }
    // simulate robot forward
    state->robot.turn(next_command.turn_rad);
    state->robot.move(next_command.move_m);

    state->map.update(state->time_of_validity);

    proto::BeaconSimDebug debug_msg;
    pack_into(state->time_of_validity, debug_msg.mutable_time_of_validity());
    pack_into(state->ekf.estimate(), debug_msg.mutable_prior());

    const liegroups::SE2 old_robot_from_new_robot = liegroups::SE2::rot(next_command.turn_rad) *
                                                    liegroups::SE2::trans(next_command.move_m, 0.0);
    pack_into(old_robot_from_new_robot, debug_msg.mutable_old_robot_from_new_robot());

    pack_into(state->ekf.predict(state->time_of_validity, old_robot_from_new_robot),
              debug_msg.mutable_prediction());

    // generate observations
    state->observations = generate_observations(state->time_of_validity, state->map, state->robot,
                                                OBS_CONFIG, make_in_out(state->gen));
    pack_into(state->observations, debug_msg.mutable_observations());

    const auto &ekf_estimate = state->ekf.update(state->observations);
    pack_into(ekf_estimate, debug_msg.mutable_posterior());
    pack_into(state->robot.local_from_robot(), debug_msg.mutable_local_from_true_robot());

    return debug_msg;
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

    WorldMap map = WorldMap(world_map_config());
    BeaconSimState state = {
        .time_of_validity = time::current_robot_time(),
        .map = map,
        .road_map = create_road_map(map),
        .robot = RobotState(INIT_POS_X_M, INIT_POS_Y_M, INIT_HEADING_RAD),
        .ekf = EkfSlam(EKF_CONFIG, time::current_robot_time()),
        .observations = {},
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
            time::SimClock::advance(DT);
            state.time_of_validity = time::current_robot_time();
            auto debug_msg = tick_sim(command, make_in_out(state), sim_config);
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

        std::this_thread::sleep_for(DT);
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
      ("autostep", "automatically step the sim")
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
        .load_off_diagonals = args["load_off_diagonals"].as<bool>(),
        .enable_brm_planner = args["enable_brm_planner"].as<bool>(),
        .autostep = args["autostep"].as<bool>(),
    });
}
