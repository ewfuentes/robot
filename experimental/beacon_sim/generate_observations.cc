
#include "experimental/beacon_sim/generate_observations.hh"

#include <cmath>

namespace robot::experimental::beacon_sim {
namespace {
double compute_range_noise(const ObservationConfig &config, InOut<std::mt19937> gen) {
    if (!config.range_noise_std_m.has_value()) {
        return 0.0;
    }

    std::normal_distribution range_noise_m(0.0, config.range_noise_std_m.value());

    return range_noise_m(*gen);
}

bool is_beyond_sensor_range(const double range_m, const std::optional<double> &max_sensor_range_m) {
    return max_sensor_range_m.has_value() && range_m > max_sensor_range_m.value();
}
}  // namespace

std::optional<BeaconObservation> generate_observation(const Beacon &beacon, const RobotState &robot,
                                                      const ObservationConfig &config,
                                                      InOut<std::mt19937> gen) {
    const Eigen::Vector2d beacon_in_robot =
        robot.local_from_robot().inverse() * beacon.pos_in_local;
    const double range_m = beacon_in_robot.norm();

    if (is_beyond_sensor_range(range_m, config.max_sensor_range_m)) {
        return std::nullopt;
    }

    const double range_noise_m = compute_range_noise(config, gen);
    const double bearing_rad = std::atan2(beacon_in_robot.y(), beacon_in_robot.x());
    return BeaconObservation{.maybe_id = beacon.id,
                             .maybe_range_m = range_m + range_noise_m,
                             .maybe_bearing_rad = bearing_rad};
}

std::vector<BeaconObservation> generate_observations(const time::RobotTimestamp &t,
                                                     const WorldMap &map, const RobotState &robot,
                                                     const ObservationConfig &config,
                                                     InOut<std::mt19937> gen) {
    std::vector<BeaconObservation> out;
    const auto beacons = map.visible_beacons(t);
    out.reserve(beacons.size());
    for (const auto &beacon : beacons) {
        const auto maybe_observation = generate_observation(beacon, robot, config, gen);
        if (maybe_observation.has_value()) {
            out.push_back(maybe_observation.value());
        }
    }
    return out;
}
}  // namespace robot::experimental::beacon_sim
