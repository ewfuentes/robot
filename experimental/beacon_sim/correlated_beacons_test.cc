
#include "experimental/beacon_sim/correlated_beacons.hh"

#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {

TEST(CorrelatedBeaconsTest, independent_beacons) {
    // Setup
    const BeaconClique clique = {.p_beacon = 0.5, .p_no_beacons = 0.25, .members = {1, 2}};

    // Action
    const BeaconPotential potential = create_correlated_beacons(clique);

    // Verification
}
}  // namespace robot::experimental::beacon_sim
