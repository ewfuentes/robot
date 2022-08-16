
#pragma once

#include "Eigen/Core"

#include <vector>

namespace experimental::beacon_sim {
struct Beacon {
    int id;
    Eigen::Vector2d pos_in_local_m;
};

struct FixedBeacons {
    std::vector<Beacon> beacons;
};

struct WorldMapOptions {
    FixedBeacons fixed_beacons;
};

class WorldMap {
   public:
    WorldMap(const WorldMapOptions &options);

    const std::vector<Beacon> &beacons() const { return beacons_; }

   private:
    std::vector<Beacon> beacons_;
};
}  // namespace experimental::beacon_sim
