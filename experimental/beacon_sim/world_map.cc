
#include "experimental/beacon_sim/world_map.hh"

#include <algorithm>
#include <chrono>
#include <random>
#include <unordered_map>

#include "common/argument_wrapper.hh"
#include "common/time/robot_time.hh"

namespace robot::experimental::beacon_sim {
namespace {

std::vector<bool> get_beacon_configuration(const CorrelatedBeaconsConfig &config,
                                           InOut<std::mt19937> gen) {
    if (config.configuration.has_value()) {
        return config.configuration.value();
    } else if (config.beacons.size() == 0) {
        return {};
    }

    const std::vector<int> present_beacon_ids = config.potential.sample(gen);
    const std::vector<int> all_members = config.potential.members();
    std::vector<bool> out(all_members.size(), false);
    for (const int beacon_id : present_beacon_ids) {
        const auto iter = std::find(all_members.begin(), all_members.end(), beacon_id);
        ROBOT_CHECK(iter != all_members.end());
        const int idx = std::distance(all_members.begin(), iter);
        out.at(idx) = true;
    }
    return out;
}
}  // namespace

WorldMap::WorldMap(const WorldMapConfig &config, const size_t seed)
    : config_(config), generator_(seed), obstacles_(config.obstacles.obstacles) {
    // Add the fixed beacons
    const int num_beacons = config.fixed_beacons.beacons.size() +
                            config.blinking_beacons.beacons.size() +
                            config.correlated_beacons.beacons.size();
    beacons_.reserve(num_beacons);

    std::transform(
        config.fixed_beacons.beacons.begin(), config.fixed_beacons.beacons.end(),
        std::back_inserter(beacons_), [](const auto &beacon) -> CompleteBeacon {
            return {.id = beacon.id,
                    .pos_in_local = beacon.pos_in_local,
                    .is_initially_visible = true,
                    .transition_times = {time::RobotTimestamp::min(), time::RobotTimestamp::max()}};
        });

    // Add the blinking beacons
    const time::RobotTimestamp init_time = time::RobotTimestamp();
    const double average_visible_time_s = 1.0 / config.blinking_beacons.beacon_disappear_rate_hz;
    const double average_invisible_time_s = 1.0 / config.blinking_beacons.beacon_appear_rate_hz;
    const double is_visible_probability =
        average_visible_time_s / (average_visible_time_s + average_invisible_time_s);
    std::transform(config.blinking_beacons.beacons.begin(), config.blinking_beacons.beacons.end(),
                   std::back_inserter(beacons_), [&](const auto &beacon) -> CompleteBeacon {
                       const bool is_initially_visible =
                           std::bernoulli_distribution(is_visible_probability)(generator_);
                       const double transition_rate_hz =
                           is_initially_visible ? config.blinking_beacons.beacon_disappear_rate_hz
                                                : config.blinking_beacons.beacon_appear_rate_hz;

                       const double transition_time_s =
                           std::exponential_distribution(transition_rate_hz)(generator_);

                       const time::RobotTimestamp::duration transition_time =
                           time::as_duration(transition_time_s);
                       return {.id = beacon.id,
                               .pos_in_local = beacon.pos_in_local,
                               .is_initially_visible = is_initially_visible,
                               .transition_times = {init_time, init_time + transition_time}};
                   });

    // Add the correlated beacons
    const std::vector<bool> configuration =
        get_beacon_configuration(config.correlated_beacons, make_in_out(generator_));

    for (int i = 0; i < static_cast<int>(configuration.size()); i++) {
        const auto &beacon = config.correlated_beacons.beacons.at(i);
        beacons_.push_back({
            .id = beacon.id,
            .pos_in_local = beacon.pos_in_local,
            .is_initially_visible = configuration.at(i),
            .transition_times = {time::RobotTimestamp::min(), time::RobotTimestamp::max()},
        });
    }
    // TODO Incorporate fixed beacons into potential
    beacon_potential_ = config.correlated_beacons.potential;
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

void WorldMap::update(const time::RobotTimestamp &t) {
    for (auto &beacon : beacons_) {
        while (t >= beacon.transition_times.back()) {
            const double transition_rate_hz =
                beacon.is_visible(t) ? config_.blinking_beacons.beacon_disappear_rate_hz
                                     : config_.blinking_beacons.beacon_appear_rate_hz;
            const double next_segment_length_s =
                std::exponential_distribution(transition_rate_hz)(generator_);
            const time::RobotTimestamp::duration next_segment_length =
                time::as_duration(next_segment_length_s);
            const time::RobotTimestamp next_transition_time =
                beacon.transition_times.back() + next_segment_length;
            beacon.transition_times.push_back(next_transition_time);
        }
    }
}

bool WorldMap::CompleteBeacon::is_visible(const time::RobotTimestamp &t) const {
    const auto iter = std::lower_bound(transition_times.begin(), transition_times.end(), t);
    const int segment_idx = std::distance(transition_times.begin(), iter);
    return segment_idx % 2 == 1 ? is_initially_visible : !is_initially_visible;
}
}  // namespace robot::experimental::beacon_sim
