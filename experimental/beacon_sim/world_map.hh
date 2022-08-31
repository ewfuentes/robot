
#pragma once

#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "Eigen/Core"
#include "common/time/robot_time.hh"

namespace robot::experimental::beacon_sim {
struct Beacon {
    // Note that these quantities come directly from the user and no effort is made
    // to ensure ids or positions are unique
    int id;
    Eigen::Vector2d pos_in_local;
};

struct FixedBeaconsConfig {
    // This config produces beacons with the given id's and locations.
    // These are always visible.
    std::vector<Beacon> beacons;
};

struct BlinkingBeaconsConfig {
    // This config produces fixed beacons that are occasionally visible.
    // The visibility of a particular beacon is modeled as a two state markov chain
    // where the transition probability at a particular time is modeled as an exponential
    // distribution with with the rates below.
    std::vector<Beacon> beacons;

    // When in the invisible state, the beacon will appear at this rate
    double beacon_appear_rate_hz;
    // When in the visible state, the beacon will disappear at this rate
    double beacon_disappear_rate_hz;
};

struct WorldMapOptions {
    FixedBeaconsConfig fixed_beacons;
    BlinkingBeaconsConfig blinking_beacons;
};

class WorldMap {
   public:
    WorldMap(const WorldMapOptions &options, std::unique_ptr<std::mt19937> generator);

    std::vector<Beacon> visible_beacons(const time::RobotTimestamp &t) const;

    void update(const time::RobotTimestamp &t);

   private:
    struct CompleteBeacon {
        int id;
        Eigen::Vector2d pos_in_local;
        bool is_initially_visible;
        std::vector<time::RobotTimestamp> transition_times;

        bool is_visible(const time::RobotTimestamp &t) const;
    };

    WorldMapOptions options_;
    std::vector<CompleteBeacon> beacons_;
    std::unique_ptr<std::mt19937> generator_;
};
}  // namespace robot::experimental::beacon_sim
