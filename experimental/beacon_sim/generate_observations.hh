
#pragma once

#include <optional>
#include <vector>

#include "experimental/beacon_sim/world_map.hh"
#include "experimental/beacon_sim/robot.hh"

namespace experimental::beacon_sim {
  struct ObservationConfig {
  };

  struct BeaconObservation {
    std::optional<int> maybe_id;
    std::optional<double> maybe_range_m;
    std::optional<double> maybe_bearing_rad;
  };

    std::optional<BeaconObservation> generate_observation(const Beacon &beacon, const RobotState &robot, const ObservationConfig &config);
    std::vector<BeaconObservation> generate_observations(const WorldMap &map, const RobotState &robot, const ObservationConfig &config);
}
