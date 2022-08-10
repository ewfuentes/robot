
#include "experimental/beacon_sim/generate_observations.hh"

#include <cmath>

namespace experimental::beacon_sim {
std::optional<BeaconObservation> generate_observation(const Beacon &beacon, const RobotState &robot,
                                                      const ObservationConfig &) {
    const double dx_m = beacon.pos_x_m - robot.pos_x_m();
    const double dy_m = beacon.pos_y_m - robot.pos_y_m();
    const double range_m = std::hypot(dx_m, dy_m);

    const double angle_rad = std::atan2(dy_m, dx_m);
    const double bearing_rad = angle_rad - robot.heading_rad();
    return BeaconObservation{
        .maybe_id = beacon.id, .maybe_range_m = range_m, .maybe_bearing_rad = bearing_rad};
}

std::vector<BeaconObservation> generate_observations(const WorldMap &map, const RobotState &robot,
                                                     const ObservationConfig &config) {
    std::vector<BeaconObservation> out;
    out.reserve(map.beacons().size());
    for (const auto &beacon : map.beacons()) {
        const auto maybe_observation = generate_observation(beacon, robot, config);
        if (maybe_observation.has_value()) {
            out.push_back(maybe_observation.value());
        }
    }
    return out;
}
}  // namespace experimental::beacon_sim
