
#pragma once

#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "Eigen/Core"
#include "common/time/robot_time.hh"
#include "experimental/beacon_sim/beacon_potential.hh"
#include "experimental/beacon_sim/obstacle.hh"

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

struct CorrelatedBeaconsConfig {
    std::vector<Beacon> beacons;
    BeaconPotential potential;
    // If configuration is specified, and the i'th entry is true, then the
    // i'th beacon is present. If the configuration isn't specified, then
    // a random configuration will be selected at map construction time.
    std::optional<std::vector<bool>> configuration;
};

struct ObstaclesConfig {
    std::vector<Obstacle> obstacles;
};

struct WorldMapConfig {
    FixedBeaconsConfig fixed_beacons;
    BlinkingBeaconsConfig blinking_beacons;
    CorrelatedBeaconsConfig correlated_beacons;
    ObstaclesConfig obstacles;
};

class WorldMap {
   public:
    WorldMap(const WorldMapConfig &config, const size_t seed = 0);

    std::vector<Beacon> visible_beacons(const time::RobotTimestamp &t) const;

    const std::vector<Obstacle> &obstacles() const { return obstacles_; }

    void update(const time::RobotTimestamp &t);

    const BeaconPotential &beacon_potential() const { return beacon_potential_; }

   private:
    struct CompleteBeacon {
        int id;
        Eigen::Vector2d pos_in_local;
        bool is_initially_visible;
        std::vector<time::RobotTimestamp> transition_times;

        bool is_visible(const time::RobotTimestamp &t) const;
    };

    WorldMapConfig config_;
    std::mt19937 generator_;
    std::vector<Obstacle> obstacles_;
    std::vector<CompleteBeacon> beacons_;
    BeaconPotential beacon_potential_;
};
}  // namespace robot::experimental::beacon_sim
