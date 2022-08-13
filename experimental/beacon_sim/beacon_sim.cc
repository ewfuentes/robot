
#include <GL/glu.h>

#include <array>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <utility>

#include "experimental/beacon_sim/generate_observations.hh"
#include "experimental/beacon_sim/robot.hh"
#include "experimental/beacon_sim/world_map.hh"
#include "visualization/gl_window/gl_window.hh"

using namespace std::literals::chrono_literals;

namespace experimental::beacon_sim {

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

RobotCommand get_command(const KeyCommand key_command) {
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
                Beacon{.id = beacon_id++, .pos_x_m = c * SPACING_M, .pos_y_m = r * SPACING_M});
        }
    }
    return out;
}

void display_state(const WorldMap &world_map, const RobotState &robot,
                   const std::vector<BeaconObservation> &observations,
                   visualization::gl_window::GlWindow &window) {
    constexpr double PX_FROM_M = 0.05;
    constexpr double BEACON_HALF_WIDTH_M = 0.25;
    constexpr double ROBOT_SIZE_M = 0.5;
    constexpr double DEG_FROM_RAD = 180.0 / std::numbers::pi;

    window.register_render_callback([=]() {
        const auto gl_error = glGetError();
        if (gl_error != GL_NO_ERROR) {
            std::cout << "GL ERROR: " << gl_error << ": " << gluErrorString(gl_error) << std::endl;
        }
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();                     // In Screen Frame
        glScaled(PX_FROM_M, PX_FROM_M, 0.0);  // In world frame

        // Draw beacons
        for (const auto &beacon : world_map.beacons()) {
            glBegin(GL_LINE_LOOP);
            glColor4f(1.0, 0.0, 0.0, 0.0);
            for (const auto &[dx, dy] :
                 std::array<std::pair<int, int>, 4>{{{-1, -1}, {-1, 1}, {1, 1}, {1, -1}}}) {
                const double x_m = beacon.pos_x_m + dx * BEACON_HALF_WIDTH_M;
                const double y_m = beacon.pos_y_m + dy * BEACON_HALF_WIDTH_M;
                glVertex2d(x_m, y_m);
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

    visualization::gl_window::GlWindow gl_window(1024, 768);

    // Initialize world map
    const WorldMap map(world_map_config());

    // Initialize robot state
    constexpr double INIT_POS_X_M = 0.0;
    constexpr double INIT_POS_Y_M = 0.0;
    constexpr double INIT_HEADING_RAD = 0.0;
    constexpr ObservationConfig OBS_CONFIG = {};
    RobotState robot(INIT_POS_X_M, INIT_POS_Y_M, INIT_HEADING_RAD);

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

    bool did_update = true;
    while (run) {
        // generate observations
        const auto observations = generate_observations(map, robot, OBS_CONFIG);

        if (did_update) {
            did_update = false;
            display_state(map, robot, observations, gl_window);
        }

        // get command
        const auto command = get_command(key_command.exchange(KeyCommand::make_reset()));
        if (command.turn_rad != 0.0 || command.move_m != 0.0) {
            did_update = true;
        }

        if (command.should_exit) {
            run = false;
        }

        // simulate robot forward
        robot.turn(command.turn_rad);
        robot.move(command.move_m);
        std::this_thread::sleep_for(25ms);
    }
}
}  // namespace experimental::beacon_sim

int main() {
    std::cout << "Hello World!" << std::endl;
    experimental::beacon_sim::run_simulation();
}
