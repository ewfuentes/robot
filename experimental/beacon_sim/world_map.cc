
#include "experimental/beacon_sim/world_map.hh"

#include <algorithm>
#include <chrono>

namespace robot::experimental::beacon_sim {
WorldMap::WorldMap(const WorldMapOptions &options, std::unique_ptr<std::mt19937> generator) {
    const int num_beacons =
        options.fixed_beacons.beacons.size() + options.blinking_beacons.beacons.size();
    beacons_.reserve(num_beacons);
    generator_ = std::move(generator);

    std::transform(
        options.fixed_beacons.beacons.begin(), options.fixed_beacons.beacons.end(),
        std::back_inserter(beacons_), [](const auto &beacon) -> CompleteBeacon {
            return {.id = beacon.id,
                    .pos_in_local = beacon.pos_in_local,
                    .is_initially_visible = true,
                    .transition_times = {time::RobotTimestamp::min(), time::RobotTimestamp::max()}};
        });

    const time::RobotTimestamp init_time = time::current_robot_time();
    const double average_visible_time_s = 1.0 / options.blinking_beacons.beacon_disappear_rate_hz;
    const double average_invisible_time_s = 1.0 / options.blinking_beacons.beacon_appear_rate_hz;
    const double is_visible_probability =
        average_visible_time_s / (average_visible_time_s + average_invisible_time_s);
    std::transform(options.blinking_beacons.beacons.begin(), options.blinking_beacons.beacons.end(),
                   std::back_inserter(beacons_), [&](const auto &beacon) -> CompleteBeacon {
                       const bool is_initially_visible =
                           std::bernoulli_distribution(is_visible_probability)(*generator_);
                       const double transition_rate_hz =
                           is_initially_visible ? options.blinking_beacons.beacon_disappear_rate_hz
                                                : options.blinking_beacons.beacon_appear_rate_hz;

                       const double transition_time_s =
                           std::exponential_distribution(transition_rate_hz)(*generator_);

                       const time::RobotTimestamp::duration transition_time =
                           time::as_duration(transition_time_s);
                       return {.id = beacon.id,
                               .pos_in_local = beacon.pos_in_local,
                               .is_initially_visible = is_initially_visible,
                               .transition_times = {init_time, init_time + transition_time}};
                   });
}

std::vector<Beacon> WorldMap::visible_beacons(const time::RobotTimestamp &t) const {
    std::vector<Beacon> out;

    for (const auto &beacon : beacons_) {
        if (beacon.is_visible(t)) {
            out.push_back({.id = beacon.id, .pos_in_local = beacon.pos_in_local});
        }
    }
    return out;
}

bool WorldMap::CompleteBeacon::is_visible(const time::RobotTimestamp &t) const {
    (void)t;
    return true;
}
}  // namespace robot::experimental::beacon_sim
