
#pragma once

#include <optional>
#include <random>
#include <vector>

#include "common/argument_wrapper.hh"
#include "experimental/beacon_sim/robot.hh"
#include "experimental/beacon_sim/world_map.hh"

namespace robot::experimental::beacon_sim {
struct ObservationConfig {
    std::optional<double> range_noise_std_m;
    std::optional<double> max_sensor_range_m;
};

struct BeaconObservation {
    std::optional<int> maybe_id;
    std::optional<double> maybe_range_m;
    std::optional<double> maybe_bearing_rad;
};

std::optional<BeaconObservation> generate_observation(const Beacon &beacon, const RobotState &robot,
                                                      const ObservationConfig &config,
                                                      InOut<std::mt19937> gen);

std::vector<BeaconObservation> generate_observations(const time::RobotTimestamp &t,
                                                     const WorldMap &map, const RobotState &robot,
                                                     const ObservationConfig &config,
                                                     InOut<std::mt19937> gen);
}  // namespace robot::experimental::beacon_sim
