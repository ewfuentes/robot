
#pragma once

#include <vector>

namespace robot::experimental::pokerbots {
struct BinCenter {
    double strength;
    double negative_potential;
    double positive_potential;
};

struct PerTurnBinCenters {
    std::vector<BinCenter> preflop_centers;
    std::vector<BinCenter> flop_centers;
    std::vector<BinCenter> turn_centers;
    std::vector<BinCenter> river_centers;
};
}  // namespace robot::experimental::pokerbots
