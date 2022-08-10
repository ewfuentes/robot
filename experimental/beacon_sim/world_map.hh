
#pragma once

#include <vector>


namespace experimental::beacon_sim {
  struct Beacon {
    double pos_x_m;
    double pos_y_m;
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

    const std::vector<Beacon> &beacons() { return beacons_; }

  private:
    std::vector<Beacon> beacons_;
  };
}
