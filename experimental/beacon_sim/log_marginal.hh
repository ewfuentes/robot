
#pragma once

#include <vector>

namespace robot::experimental::beacon_sim {

struct LogMarginal {
    std::vector<int> present_beacons;
    double log_marginal;
};

}  // namespace robot::experimental::beacon_sim
