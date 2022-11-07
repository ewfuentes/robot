
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
#include "common/proto/load_from_file.hh"
#include "common/time/robot_time_to_proto.hh"
#include "common/time/sim_clock.hh"
#include "cxxopts.hpp"
#include "experimental/beacon_sim/beacon_observation_to_proto.hh"
#include "experimental/beacon_sim/beacon_sim_debug.pb.h"
#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/ekf_slam_estimate_to_proto.hh"
#include "experimental/beacon_sim/extract_mapped_landmarks.hh"
#include "experimental/beacon_sim/generate_observations.hh"
#include "experimental/beacon_sim/mapped_landmarks_to_proto.hh"
#include "experimental/beacon_sim/robot.hh"
#include "experimental/beacon_sim/sim_log.pb.h"
#include "experimental/beacon_sim/world_map.hh"
#include "planning/probabilistic_road_map.hh"
#include "visualization/gl_window/gl_window.hh"

using namespace std::literals::chrono_literals;

namespace robot::experimental::beacon_sim {
namespace {

struct SimConfig {
    std::optional<std::string> log_path;
    std::optional<std::string> map_input_path;
    std::optional<std::string> map_output_path;
    bool load_off_diagonals;
    bool autostep;
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
    bool should_print_cov;
};

struct KeyCommand {
    bool arrow_left;
    bool arrow_right;
    bool arrow_up;
    bool arrow_down;
    bool q;
    bool p;

    static KeyCommand make_reset() {
        return KeyCommand{
            .arrow_left = false,
            .arrow_right = false,
            .arrow_up = false,
            .arrow_down = false,
            .q = false,
            .p = false,
        };
    }
};

RobotCommand get_command(
    const KeyCommand key_command,
    const std::unordered_map<int, visualization::gl_window::JoystickState> &joysticks) {
    constexpr double TURN_AMOUNT_RAD = std::numbers::pi / 4.0;
    constexpr double MOVE_AMOUNT_M = 0.1;
    // Handle Keyboard Commands
    RobotCommand out = {
        .turn_rad = 0,
        .move_m = 0,
        .should_exit = false,
        .should_step = false,
        .should_print_cov = false,
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

void display_state(const time::RobotTimestamp &t, const WorldMap &world_map,
                   const RobotState &robot, const std::vector<BeaconObservation> &observations,
                   const EkfSlamEstimate &ekf_estimate, const planning::RoadMap &road_map,
                   InOut<visualization::gl_window::GlWindow> window) {
    (void)road_map;
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

            const Eigen::Matrix3d pos_cov = ekf_estimate.robot_cov();
            const Eigen::LLT<Eigen::Matrix3d> cov_llt(pos_cov);

            glBegin(GL_LINE_LOOP);
            glColor4f(1.0, 0.5, 0.5, 1.0);
            for (double theta = 0.0; theta <= 2 * std::numbers::pi; theta += 0.005) {
                const Eigen::Vector3d tangent_vec =
                    cov_llt.matrixL() *
                    Eigen::Vector3d{2.0 * std::cos(theta), 2.0 * std::sin(theta), 0.0};
                const liegroups::SE2 local_from_ellipse_pt =
                    ekf_estimate.local_from_robot() * liegroups::SE2::exp(tangent_vec);
                const Eigen::Vector2d pt = local_from_ellipse_pt.translation();

                glVertex2d(pt.x(), pt.y());
            }
            glEnd();
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
            glPopMatrix();  // Pop from beacon frame to world frame
        }

        // Draw Road map
        int num_points = road_map.points.size();
        for (int i = 0; i < num_points; i++) {
            // Draw the node
            const Eigen::Vector2d &pt = road_map.points.at(i);
            glPushMatrix();
            const liegroups::SE3 local_from_node = se3_from_se2(liegroups::SE2::trans(pt));
            glMultMatrixd(local_from_node.matrix().data());
            glBegin(GL_LINE_LOOP);
            glColor3f(0.0, 0.5, 0.5);
            for (const auto &corner : std::array<Eigen::Vector2d, 4>{
                     {{-1.0, -1.0}, {-1.0, 1.0}, {1.0, 1.0}, {1.0, -1.0}}}) {
                constexpr double NODE_HALF_WIDTH_M = 0.25 / 2.0;
                const Eigen::Vector2d corner_in_node = corner * NODE_HALF_WIDTH_M;
                glVertex2d(corner_in_node.x(), corner_in_node.y());
            }
            glEnd();

            glPopMatrix();  // Pop from road map node frame to world frame

            for (int j = i + 1; j < num_points; j++) {
                if (road_map.adj(i, j)) {
                    // Draw an edge between the two points
                    const Eigen::Vector2d other_pt = road_map.points.at(j);
                    glColor3f(0.4, 0.4, 0.4);
                    glBegin(GL_LINES);
                    glVertex2d(pt.x(), pt.y());
                    glVertex2d(other_pt.x(), other_pt.y());
                    glEnd();
                }
            }
        }
    });
}

planning::RoadMap create_road_map(const WorldMap &map) {
    const planning::RoadmapCreationConfig config = {
        .seed = 0,
        .num_valid_points = 60,
        .max_node_degree = 5,
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

    std::mt19937 gen(0);

    visualization::gl_window::GlWindow gl_window(1280, 960);

    // Initialize world map
    WorldMap map(world_map_config(), std::make_unique<std::mt19937>(0));

    time::set_default_time_provider(time::TimeProvider::SIM);

    // Initialize robot state
    constexpr double INIT_POS_X_M = 0.0;
    constexpr double INIT_POS_Y_M = 0.0;
    constexpr double INIT_HEADING_RAD = 0.0;
    constexpr ObservationConfig OBS_CONFIG = {
        .range_noise_std_m = 0.1,
        .max_sensor_range_m = 5.0,
    };
    RobotState robot(INIT_POS_X_M, INIT_POS_Y_M, INIT_HEADING_RAD);

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
    };
    EkfSlam ekf_slam(EKF_CONFIG, time::current_robot_time());

    if (sim_config.map_input_path) {
        load_map(sim_config, make_in_out(ekf_slam));
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
            }
            key_command.store(update);
        });

    const auto DT = 25ms;
    std::vector<proto::BeaconSimDebug> debug_msgs;
    debug_msgs.reserve(10000);

    const planning::RoadMap road_map = create_road_map(map);

    while (run) {
        // get command
        const auto command = get_command(key_command.exchange(KeyCommand::make_reset()),
                                         gl_window.get_joystick_states());

        if (command.should_exit) {
            run = false;
        }
        if (command.should_print_cov) {
            // Print out the covariance for each beacon
            std::cout << "************************************************ " << std::endl;
            const auto &ekf_estimate = ekf_slam.estimate();
            for (const int beacon_id : ekf_estimate.beacon_ids) {
                std::cout << "================== Beacon: " << beacon_id << std::endl;
                std::cout << ekf_estimate.beacon_cov(beacon_id).value() << std::endl;
            }
        }

        if (command.should_step || sim_config.autostep) {
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

            pack_into(ekf_slam.predict(time::current_robot_time(), old_robot_from_new_robot),
                      debug_msg.mutable_prediction());

            // generate observations
            const auto observations = generate_observations(time::current_robot_time(), map, robot,
                                                            OBS_CONFIG, make_in_out(gen));
            pack_into(observations, debug_msg.mutable_observations());

            const auto &ekf_estimate = ekf_slam.update(observations);
            pack_into(ekf_estimate, debug_msg.mutable_posterior());
            pack_into(robot.local_from_robot(), debug_msg.mutable_local_from_true_robot());

            display_state(time::current_robot_time(), map, robot, observations, ekf_estimate,
                          road_map, make_in_out(gl_window));
            debug_msgs.emplace_back(std::move(debug_msg));
        }

        std::this_thread::sleep_for(DT);
    }

    if (sim_config.log_path) {
        write_out_log_file(sim_config, std::move(debug_msgs));
    }
    if (sim_config.map_output_path) {
        write_out_map(sim_config, ekf_slam.estimate());
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
        .autostep = args["autostep"].as<bool>(),
    });
}
