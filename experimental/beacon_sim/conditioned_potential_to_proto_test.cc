
#include "experimental/beacon_sim/conditioned_potential_to_proto.hh"

#include "experimental/beacon_sim/beacon_potential.hh"
#include "experimental/beacon_sim/correlated_beacons.hh"
#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {

TEST(ConditionedPotentialToProtoTest, pack_unpack) {
    // Setup
    const BeaconClique clique{.p_beacon = 0.5, .p_no_beacons = 0.1, .members = {2, 3, 4, 5}};
    const BeaconPotential underlying = create_correlated_beacons(clique);
    const BeaconPotential conditioned = underlying.conditioned_on({{2, true}, {3, false}});

    // Action
    proto::BeaconPotential proto;
    pack_into(conditioned, &proto);
    const BeaconPotential unpacked = unpack_from(proto);

    // Verification
    const std::vector<std::unordered_map<int, bool>> assignments = {
        {{4, false}, {5, false}},
        {{4, false}, {5, true}},
        {{4, true}, {5, false}},
        {{4, true}, {5, true}},
    };

    const bool DONT_ALLOW_PARTIAL_ASSIGNMENTS = false;
    constexpr double TOL = 1e-6;
    for (const auto &assignment : assignments) {
        EXPECT_NEAR(conditioned.log_prob(assignment, DONT_ALLOW_PARTIAL_ASSIGNMENTS),
                    unpacked.log_prob(assignment, DONT_ALLOW_PARTIAL_ASSIGNMENTS), TOL);
    }
}
}  // namespace robot::experimental::beacon_sim
