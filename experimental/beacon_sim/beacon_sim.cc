
#include <GL/glu.h>

#include <array>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <utility>

#include "experimental/beacon_sim/generate_observations.hh"
#include "experimental/beacon_sim/robot.hh"
#include "experimental/beacon_sim/sim_clock.hh"
#include "experimental/beacon_sim/world_map.hh"
#include "visualization/gl_window/gl_window.hh"

using namespace std::literals::chrono_literals;

namespace robot::experimental::beacon_sim {

struct RobotCommand {
    double turn_rad;
    double move_m;
    bool should_exit;
};

struct KeyCommand {
    bool arrow_left;
    bool arrow_right;
    bool arrow_up;
    bool arrow_down;
    bool q;

    static KeyCommand make_reset() {
        return KeyCommand{.arrow_left = false,
                          .arrow_right = false,
                          .arrow_up = false,
                          .arrow_down = false,
                          .q = false};
    }
};

RobotCommand get_command(
    const KeyCommand key_command,
    const std::unordered_map<int, visualization::gl_window::JoystickState> &joysticks) {
    // Handle Keyboard Commands
    if (key_command.q) {
        return {.turn_rad = 0, .move_m = 0, .should_exit = true};
    } else if (key_command.arrow_left) {
        return {.turn_rad = std::numbers::pi / 4.0, .move_m = 0, .should_exit = false};
    } else if (key_command.arrow_right) {
        return {.turn_rad = -std::numbers::pi / 4.0, .move_m = 0, .should_exit = false};
    } else if (key_command.arrow_up) {
        return {.turn_rad = 0.0, .move_m = 0.1, .should_exit = false};
    } else if (key_command.arrow_down) {
        return {.turn_rad = 0.0, .move_m = -0.1, .should_exit = false};
    }

    for (const auto &[id, state] : joysticks) {
        if (state.buttons[GLFW_GAMEPAD_BUTTON_CIRCLE] == GLFW_PRESS) {
            return {.turn_rad = 0, .move_m = 0, .should_exit = true};
        }
        const double left = -state.axes[GLFW_GAMEPAD_AXIS_LEFT_Y];
        const double right = -state.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y];

        const double mean = (left + right) / 2.0;
        const double mean_diff = (right - left) / 2.0;

        return {.turn_rad = 0.1 * mean_diff, .move_m = mean * 0.1, .should_exit = false};
    }
    return {.turn_rad = 0, .move_m = 0, .should_exit = false};
}

WorldMapOptions world_map_config() {
    // Create a grid of beacons
    constexpr int NUM_ROWS = 4;
    constexpr int NUM_COLS = 5;
    constexpr double SPACING_M = 3.0;
    WorldMapOptions out;
    out.fixed_beacons.beacons.reserve(NUM_ROWS * NUM_COLS);
    int beacon_id = 0;
    for (int r = 0; r < NUM_ROWS; r++) {
        for (int c = 0; c < NUM_COLS; c++) {
            out.fixed_beacons.beacons.push_back(
                Beacon{.id = beacon_id++, .pos_in_local = {c * SPACING_M, r * SPACING_M}});
        }
    }
    return out;
}

void display_state(const WorldMap &world_map, const RobotState &robot,
                   const std::vector<BeaconObservation> &observations,
                   visualization::gl_window::GlWindow &window) {
    constexpr double BEACON_HALF_WIDTH_M = 0.25;
    constexpr double ROBOT_SIZE_M = 0.5;
    constexpr double DEG_FROM_RAD = 180.0 / std::numbers::pi;
    constexpr double WINDOW_WIDTH_M = 15;
    const auto [screen_width_px, screen_height_px] = window.get_window_dims();
    const double aspect_ratio = static_cast<double>(screen_height_px) / screen_width_px;

    window.register_render_callback([=]() {
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
        for (const auto &beacon : world_map.beacons()) {
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
    });
}

void run_simulation() {
    bool run = true;

    std::mt19937 gen(0);

    visualization::gl_window::GlWindow gl_window(1024, 768);

    // Initialize world map
    const WorldMap map(world_map_config());

    // Initialize robot state
    constexpr double INIT_POS_X_M = 0.0;
    constexpr double INIT_POS_Y_M = 0.0;
    constexpr double INIT_HEADING_RAD = 0.0;
    constexpr ObservationConfig OBS_CONFIG = {
        .range_noise_std_m = 0.5,
    };
    RobotState robot(INIT_POS_X_M, INIT_POS_Y_M, INIT_HEADING_RAD);

    gl_window.register_window_resize_callback(
        [](const int width, const int height) { glViewport(0, 0, width, height); });

    std::atomic<KeyCommand> key_command;
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

    SimClock::reset();
    const auto DT = 25ms;
    while (run) {
        // generate observations
        const auto observations = generate_observations(map, robot, OBS_CONFIG, make_in_out(gen));

        display_state(map, robot, observations, gl_window);

        // get command
        const auto command = get_command(key_command.exchange(KeyCommand::make_reset()),
                                         gl_window.get_joystick_states());

        if (command.should_exit) {
            run = false;
        }

        // simulate robot forward
        robot.turn(command.turn_rad);
        robot.move(command.move_m);
        std::this_thread::sleep_for(DT);
        SimClock::advance(DT);
    }
}
}  // namespace robot::experimental::beacon_sim

int main() {
    std::cout << "Hello World!" << std::endl;
    robot::experimental::beacon_sim::run_simulation();
}
