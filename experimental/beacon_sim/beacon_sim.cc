
#include <GL/gl.h>
#include <GL/glu.h>

#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <utility>

#include "Eigen/Cholesky"
#include "common/argument_wrapper.hh"
#include "common/liegroups/se2_to_proto.hh"
#include "common/liegroups/se3.hh"
#include "common/time/robot_time_to_proto.hh"
#include "common/time/sim_clock.hh"
#include "cxxopts.hpp"
#include "experimental/beacon_sim/beacon_observation_to_proto.hh"
#include "experimental/beacon_sim/beacon_sim_debug.pb.h"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/ekf_slam_estimate_to_proto.hh"
#include "experimental/beacon_sim/generate_observations.hh"
#include "experimental/beacon_sim/robot.hh"
#include "experimental/beacon_sim/sim_log.pb.h"
#include "experimental/beacon_sim/world_map.hh"
#include "visualization/gl_window/gl_window.hh"

using namespace std::literals::chrono_literals;

namespace robot::experimental::beacon_sim {
namespace {

struct SimConfig {
    std::string log_path;
};

liegroups::SE3 se3_from_se2(const liegroups::SE2 &a_from_b) {
    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d a_from_b_mat = a_from_b.matrix();
    mat.topLeftCorner(2, 2) = a_from_b_mat.topLeftCorner(2, 2);
    mat.topRightCorner(2, 1) = a_from_b_mat.topRightCorner(2, 1);
    return liegroups::SE3(mat);
}
}  // namespace

struct RobotCommand {
    double turn_rad;
    double move_m;
    bool should_exit;
    bool should_step;
};

struct KeyCommand {
    bool arrow_left;
    bool arrow_right;
    bool arrow_up;
    bool arrow_down;
    bool q;

    static KeyCommand make_reset() {
        return KeyCommand{
            .arrow_left = false,
            .arrow_right = false,
            .arrow_up = false,
            .arrow_down = false,
            .q = false,
        };
    }
};

RobotCommand get_command(
    const KeyCommand key_command,
    const std::unordered_map<int, visualization::gl_window::JoystickState> &joysticks) {
    // Handle Keyboard Commands
    if (key_command.q) {
        return {.turn_rad = 0, .move_m = 0, .should_exit = true, .should_step = true};
    } else if (key_command.arrow_left) {
        return {.turn_rad = std::numbers::pi / 4.0,
                .move_m = 0,
                .should_exit = false,
                .should_step = true};
    } else if (key_command.arrow_right) {
        return {.turn_rad = -std::numbers::pi / 4.0,
                .move_m = 0,
                .should_exit = false,
                .should_step = true};
    } else if (key_command.arrow_up) {
        return {.turn_rad = 0.0, .move_m = 0.1, .should_exit = false, .should_step = true};
    } else if (key_command.arrow_down) {
        return {.turn_rad = 0.0, .move_m = -0.1, .should_exit = false, .should_step = true};
    }

    for (const auto &[id, state] : joysticks) {
        if (state.buttons[GLFW_GAMEPAD_BUTTON_CIRCLE] == GLFW_PRESS) {
            return {.turn_rad = 0, .move_m = 0, .should_exit = true};
        }
        const double left = -state.axes[GLFW_GAMEPAD_AXIS_LEFT_Y];
        const double right = -state.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y];

        const double mean = (left + right) / 2.0;
        const double mean_diff = (right - left) / 2.0;

        return {.turn_rad = 0.1 * mean_diff,
                .move_m = mean * 0.1,
                .should_exit = false,
                .should_step = true};
    }
    return {.turn_rad = 0, .move_m = 0, .should_exit = false, .should_step = false};
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
    // Create a grid of beacons
    constexpr int NUM_ROWS = 2;
    constexpr int NUM_COLS = 2;
    constexpr double SPACING_M = 4.0;
    WorldMapConfig out;
    out.fixed_beacons.beacons.reserve(NUM_ROWS * NUM_COLS);
    int beacon_id = 0;
    for (int r = 0; r < NUM_ROWS; r++) {
        for (int c = 0; c < NUM_COLS; c++) {
            out.fixed_beacons.beacons.push_back(
                Beacon{.id = beacon_id++, .pos_in_local = {c * SPACING_M, r * SPACING_M}});
        }
    }
    out.blinking_beacons = {
        .beacons = {},
        .beacon_appear_rate_hz = 1.0,
        .beacon_disappear_rate_hz = 0.5,
    };

    out.blinking_beacons.beacons.reserve(NUM_ROWS * NUM_COLS);

    for (int r = 0; r < NUM_ROWS; r++) {
        for (int c = 0; c < NUM_COLS; c++) {
            out.blinking_beacons.beacons.push_back(
                Beacon{.id = beacon_id++, .pos_in_local = {-c * SPACING_M, -r * SPACING_M}});
        }
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

void display_state(const time::RobotTimestamp &t, const WorldMap &world_map,
                   const RobotState &robot, const std::vector<BeaconObservation> &observations,
                   const EkfSlamEstimate &ekf_estimate,
                   InOut<visualization::gl_window::GlWindow> window) {
    constexpr double BEACON_HALF_WIDTH_M = 0.25;
    constexpr double ROBOT_SIZE_M = 0.5;
    constexpr double DEG_FROM_RAD = 180.0 / std::numbers::pi;
    constexpr double WINDOW_WIDTH_M = 15;
    const auto [screen_width_px, screen_height_px] = window->get_window_dims();
    const double aspect_ratio = static_cast<double>(screen_height_px) / screen_width_px;

    window->register_render_callback([=, beacons = world_map.visible_beacons(t),
                                      obstacles = world_map.obstacles()]() {
        const auto gl_error = glGetError();
        if (gl_error != GL_NO_ERROR) {
            std::cout << "GL ERROR: " << gl_error << ": " << gluErrorString(gl_error) << std::endl;
        }
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-WINDOW_WIDTH_M, WINDOW_WIDTH_M, -WINDOW_WIDTH_M * aspect_ratio,
                WINDOW_WIDTH_M * aspect_ratio, -1.0, 1.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        // Draw beacons
        for (const auto &beacon : beacons) {
            glBegin(GL_LINE_LOOP);
            glColor4f(1.0, 0.0, 0.0, 0.0);
            for (const auto &corner : std::array<Eigen::Vector2d, 4>{
                     {{-1.0, -1.0}, {-1.0, 1.0}, {1.0, 1.0}, {1.0, -1.0}}}) {
                const Eigen::Vector2d point_in_local =
                    beacon.pos_in_local + corner * BEACON_HALF_WIDTH_M;
                glVertex2d(point_in_local.x(), point_in_local.y());
            }
            glEnd();
        }

        // Draw robot
        glPushMatrix();
        glTranslated(robot.pos_x_m(), robot.pos_y_m(), 0.0);
        glRotated(DEG_FROM_RAD * robot.heading_rad(), 0.0, 0.0, 1.0);
        glBegin(GL_LINE_LOOP);
        glColor4f(0.5, 0.5, 1.0, 1.0);
        for (const auto &[dx, dy] :
             std::array<std::pair<double, double>, 3>{{{0.0, 0.5}, {1.5, 0.0}, {0.0, -0.5}}}) {
            const double x_in_robot_m = dx * ROBOT_SIZE_M;
            const double y_in_robot_m = dy * ROBOT_SIZE_M;
            glVertex2d(x_in_robot_m, y_in_robot_m);
        }
        glEnd();

        for (const auto &obs : observations) {
            glPushMatrix();
            glRotated(DEG_FROM_RAD * obs.maybe_bearing_rad.value(), 0.0, 0.0, 1.0);
            glBegin(GL_LINES);
            glColor4f(0.5, 1.0, 0.5, 1.0);
            glVertex2d(0.0, 0.0);
            glVertex2d(obs.maybe_range_m.value(), 0.0);
            glEnd();
            glPopMatrix();  // Pop from Measurement Frame to robot frame
        }

        glPopMatrix();  // Pop from Robot Frame to world frame

        // Draw Obstacles
        {
            for (const auto &obstacle : obstacles) {
                glBegin(GL_LINE_LOOP);
                if (obstacle.is_inside(robot.local_from_robot().translation())) {
                    glColor4ub(168, 50, 50, 255);
                } else {
                    glColor4ub(124, 187, 235, 255);
                }
                for (const Eigen::Vector2d &pt : obstacle.pts_in_frame()) {
                    glVertex2d(pt.x(), pt.y());
                }
                glEnd();
            }
        }

        // Draw ekf estimates
        {
            glPushMatrix();
            const liegroups::SE3 est_local_from_robot =
                se3_from_se2(ekf_estimate.local_from_robot());
            glMultMatrixd(est_local_from_robot.matrix().data());

            glBegin(GL_LINE_LOOP);
            glColor4f(0.75, 0.75, 1.0, 1.0);
            for (const auto &[dx, dy] :
                 std::array<std::pair<double, double>, 3>{{{0.0, 0.5}, {1.5, 0.0}, {0.0, -0.5}}}) {
                const double x_in_robot_m = dx * ROBOT_SIZE_M;
                const double y_in_robot_m = dy * ROBOT_SIZE_M;
                glVertex2d(x_in_robot_m, y_in_robot_m);
            }
            glEnd();
            glPopMatrix();  // Pop from estimated robot frame to world frame

            glPushMatrix();
            // Translate to the robot origin, but throw away the rotation
            glMultMatrixd(
                liegroups::SE3::trans(est_local_from_robot.translation()).matrix().data());
            const Eigen::Matrix3d pos_cov = ekf_estimate.robot_cov();
            const Eigen::Matrix3d pos_cov_in_local =
                ekf_estimate.local_from_robot().Adj() * pos_cov *
                ekf_estimate.local_from_robot().Adj().transpose();
            const Eigen::LLT<Eigen::Matrix2d> cov_llt(pos_cov_in_local.topLeftCorner(2, 2));

            glBegin(GL_LINE_LOOP);
            glColor4f(1.0, 0.5, 0.5, 1.0);
            for (double theta = 0; theta < 2 * std::numbers::pi; theta += 0.05) {
                const Eigen::Vector2d pt =
                    cov_llt.matrixL() *
                    Eigen::Vector2d{2.0 * std::cos(theta), 2.0 * std::sin(theta)};
                glVertex2d(pt.x(), pt.y());
            }
            glEnd();
            glPopMatrix();  // Pop from robot location to world frame
        }

        for (const auto beacon_id : ekf_estimate.beacon_ids) {
            glPushMatrix();
            const liegroups::SE3 local_from_beacon = se3_from_se2(
                liegroups::SE2::trans(ekf_estimate.beacon_in_local(beacon_id).value()));
            glMultMatrixd(local_from_beacon.matrix().data());
            const Eigen::Matrix2d pos_cov = ekf_estimate.beacon_cov(beacon_id).value();
            const Eigen::LLT<Eigen::Matrix2d> cov_llt(pos_cov);

            glBegin(GL_LINE_LOOP);
            glColor4f(0.75, 0.75, 1.0, 1.0);
            for (double theta = 0; theta < 2 * std::numbers::pi; theta += 0.05) {
                const Eigen::Vector2d pt =
                    cov_llt.matrixL() *
                    Eigen::Vector2d{2.0 * std::cos(theta), 2.0 * std::sin(theta)};
                glVertex2d(pt.x(), pt.y());
            }
            glEnd();
            glPopMatrix();
        }
    });
}

void write_out_log_file(const SimConfig &sim_config,
                        std::vector<proto::BeaconSimDebug> debug_msgs) {
    proto::SimLog log_proto;
    for (proto::BeaconSimDebug &msg : debug_msgs) {
        log_proto.mutable_debug_msgs()->Add(std::move(msg));
    }
    std::fstream file_out(sim_config.log_path, std::ios::binary | std::ios::out | std::ios::trunc);
    log_proto.SerializeToOstream(&file_out);
}

void run_simulation(const SimConfig &sim_config) {
    bool run = true;

    std::mt19937 gen(0);

    visualization::gl_window::GlWindow gl_window(1024, 768);

    // Initialize world map
    WorldMap map(world_map_config(), std::make_unique<std::mt19937>(0));

    // Initialize robot state
    constexpr double INIT_POS_X_M = 0.0;
    constexpr double INIT_POS_Y_M = 0.0;
    constexpr double INIT_HEADING_RAD = 0.0;
    constexpr ObservationConfig OBS_CONFIG = {
        .range_noise_std_m = 0.1,
        .max_sensor_range_m = 10.0,
    };
    RobotState robot(INIT_POS_X_M, INIT_POS_Y_M, INIT_HEADING_RAD);

    constexpr EkfSlamConfig EKF_CONFIG = {
        .max_num_beacons = 50,
        .initial_beacon_uncertainty_m = 100,
        .along_track_process_noise_m_per_rt_meter = 1e-2,
        .cross_track_process_noise_m_per_rt_meter = 1e-9,
        .pos_process_noise_m_per_rt_s = 1e-9,
        .heading_process_noise_rad_per_rt_meter = 1e-6,
        .heading_process_noise_rad_per_rt_s = 1e-10,
        .beacon_pos_process_noise_m_per_rt_s = 0.000001,
        .range_measurement_noise_m = 0.25,
        .bearing_measurement_noise_rad = 0.01,
    };
    EkfSlam ekf_slam(EKF_CONFIG);

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
            }
            key_command.store(update);
        });

    time::SimClock::reset();
    const auto DT = 25ms;
    std::vector<proto::BeaconSimDebug> debug_msgs;
    debug_msgs.reserve(10000);
    while (run) {
        // get command
        const auto command = get_command(key_command.exchange(KeyCommand::make_reset()),
                                         gl_window.get_joystick_states());

        if (command.should_exit) {
            run = false;
        }

        if (command.should_step) {
            // simulate robot forward
            robot.turn(command.turn_rad);
            robot.move(command.move_m);

            time::SimClock::advance(DT);

            map.update(time::current_robot_time());

            proto::BeaconSimDebug debug_msg;
            pack_into(time::current_robot_time(), debug_msg.mutable_time_of_validity());
            pack_into(ekf_slam.estimate(), debug_msg.mutable_prior());

            const liegroups::SE2 old_robot_from_new_robot =
                liegroups::SE2::trans(command.move_m, 0.0) * liegroups::SE2::rot(command.turn_rad);
            pack_into(old_robot_from_new_robot, debug_msg.mutable_old_robot_from_new_robot());

            pack_into(ekf_slam.predict(old_robot_from_new_robot), debug_msg.mutable_prediction());

            // generate observations
            const auto observations = generate_observations(time::current_robot_time(), map, robot,
                                                            OBS_CONFIG, make_in_out(gen));
            pack_into(observations, debug_msg.mutable_observations());

            const auto &ekf_estimate = ekf_slam.update(observations);
            pack_into(ekf_estimate, debug_msg.mutable_posterior());
            pack_into(robot.local_from_robot(), debug_msg.mutable_local_from_true_robot());

            display_state(time::current_robot_time(), map, robot, observations, ekf_estimate,
                          make_in_out(gl_window));
            debug_msgs.emplace_back(std::move(debug_msg));
        }

        std::this_thread::sleep_for(DT);
    }

    write_out_log_file(sim_config, std::move(debug_msgs));
}
}  // namespace robot::experimental::beacon_sim

int main(int argc, char **argv) {
    const std::string DEFAULT_LOG_LOCATION = "/tmp/beacon_sim.pb";
    cxxopts::Options options("beacon_sim", "Simple Localization Simulator with Beacons");
    options.add_options()("log_file", "Path to output file.",
                          cxxopts::value<std::string>()->default_value(DEFAULT_LOG_LOCATION))(
        "help", "Print usage");

    auto args = options.parse(argc, argv);
    if (args.count("help")) {
        std::cout << options.help() << std::endl;
        std::exit(0);
    }
    robot::experimental::beacon_sim::run_simulation({
        .log_path = args["log_file"].as<std::string>(),
    });
}
