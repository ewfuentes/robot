
#include "experimental/beacon_sim/generate_observations.hh"

#include <cmath>

namespace experimental::beacon_sim {
std::optional<BeaconObservation> generate_observation(const Beacon &beacon, const RobotState &robot,
                                                      const ObservationConfig &) {
    const double range_m = (robot.local_from_robot().translation() - beacon.pos_in_local_m).norm();

    const Eigen::Vector2d robot_from_beacon =
        robot.local_from_robot().inverse() * beacon.pos_in_local_m;
    const double bearing_rad = std::atan2(robot_from_beacon.y(), robot_from_beacon.x());
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
