
#include <curses.h>

#include <iostream>

#include "experimental/beacon_sim/generate_observations.hh"
#include "experimental/beacon_sim/robot.hh"
#include "experimental/beacon_sim/world_map.hh"

namespace experimental::beacon_sim {

struct RobotCommand {
    double turn_rad;
    double move_m;
    bool should_exit;
};

RobotCommand get_command() {
    const int ch = getch();
    if (ch == ERR) {
        return {.turn_rad = 0, .move_m = 0, .should_exit = false};
    } else if (ch == 'q') {
        return {.turn_rad = 0, .move_m = 0, .should_exit = true};
    } else if (ch == KEY_LEFT) {
        return {.turn_rad = std::numbers::pi / 4.0, .move_m = 0, .should_exit = false};
    } else if (ch == KEY_RIGHT) {
        return {.turn_rad = -std::numbers::pi / 4.0, .move_m = 0, .should_exit = false};
    } else if (ch == KEY_UP) {
        return {.turn_rad = 0.0, .move_m = 0.1, .should_exit = false};
    } else if (ch == KEY_DOWN) {
        return {.turn_rad = 0.0, .move_m = -0.1, .should_exit = false};
    }

    return {.turn_rad = 0, .move_m = 0, .should_exit = false};
}

WorldMapOptions world_map_config() {
    // Create a grid of beacons
    constexpr int NUM_ROWS = 2;
    constexpr int NUM_COLS = 3;
    constexpr double SPACING_M = 3.0;
    WorldMapOptions out;
    out.fixed_beacons.beacons.reserve(NUM_ROWS * NUM_COLS);
    int beacon_id = 0;
    for (int r = 0; r < NUM_ROWS; r++) {
        for (int c = 0; c < NUM_COLS; c++) {
            out.fixed_beacons.beacons.push_back(
                Beacon{.id = beacon_id++, .pos_x_m = r * SPACING_M, .pos_y_m = c * SPACING_M});
        }
    }
    return out;
}

void display_state(const WorldMap &world_map, const RobotState &robot,
                   const std::vector<BeaconObservation> &observations) {
    clear();
    printw("Press `q` to quit\n");
    printw("Num Beacons: %zu RobotState: (%f, %f, %f) Num Observations: %zu\n",
           world_map.beacons().size(), robot.pos_x_m(), robot.pos_y_m(), robot.heading_rad(),
           observations.size());
    printw("Observations\n");
    for (const auto &obs : observations) {
        printw("ID: %d Range: %f Bearing: %f\n", obs.maybe_id.value(), obs.maybe_range_m.value(),
               obs.maybe_bearing_rad.value());
    }
}

void run_simulation() {
    bool run = true;

    // Initialize world map
    const WorldMap map(world_map_config());

    // Initialize robot state
    constexpr double INIT_POS_X_M = 0.0;
    constexpr double INIT_POS_Y_M = 0.0;
    constexpr double INIT_HEADING_RAD = 0.0;
    constexpr ObservationConfig OBS_CONFIG = {};
    RobotState robot(INIT_POS_X_M, INIT_POS_Y_M, INIT_HEADING_RAD);

    bool did_update = true;
    auto *window = initscr();
    noecho();
    cbreak();
    nodelay(window, true);
    keypad(window, true);
    while (run) {
        // generate observations
        const auto observations = generate_observations(map, robot, OBS_CONFIG);

        if (did_update) {
            did_update = false;
            display_state(map, robot, observations);
        }

        // get command
        const auto command = get_command();
        if (command.turn_rad != 0.0 || command.move_m != 0.0) {
            did_update = true;
        }

        if (command.should_exit) {
            run = false;
        }

        // simulate robot forward
        robot.turn(command.turn_rad);
        robot.move(command.move_m);
    }
    nocbreak();
    echo();
    endwin();
}
}  // namespace experimental::beacon_sim

int main() {
    std::cout << "Hello World!" << std::endl;
    experimental::beacon_sim::run_simulation();
}
