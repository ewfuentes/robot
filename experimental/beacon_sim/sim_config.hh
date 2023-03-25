
#pragma once

#include <optional>
#include <string>

#include "common/time/robot_time.hh"

namespace robot::experimental::beacon_sim {
struct SimConfig {
    std::optional<std::string> log_path;
    std::optional<std::string> map_input_path;
    std::optional<std::string> map_output_path;
    time::RobotTimestamp::duration dt;
    bool load_off_diagonals;
    bool enable_brm_planner;
    bool autostep;
    // Should this be here?
    std::optional<int> correlated_beacons_configuration;
};
}  // namespace robot::experimental::beacon_sim
