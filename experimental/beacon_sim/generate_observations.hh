
#pragma once

#include <optional>
#include <random>
#include <vector>

#include "experimental/beacon_sim/robot.hh"
#include "experimental/beacon_sim/world_map.hh"

namespace experimental::beacon_sim {
struct ObservationConfig {
    std::optional<double> range_noise_std_m;
};

struct BeaconObservation {
    std::optional<int> maybe_id;
    std::optional<double> maybe_range_m;
    std::optional<double> maybe_bearing_rad;
};

std::optional<BeaconObservation> generate_observation(const Beacon &beacon, const RobotState &robot,
                                                      std::mt19937 &gen,
                                                      const ObservationConfig &config);
std::vector<BeaconObservation> generate_observations(const WorldMap &map, const RobotState &robot,
                                                     std::mt19937 &gen,
                                                     const ObservationConfig &config);
}  // namespace experimental::beacon_sim
