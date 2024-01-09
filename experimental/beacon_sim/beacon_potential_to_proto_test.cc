
#include "experimental/beacon_sim/beacon_potential_to_proto.hh"

#include "experimental/beacon_sim/correlated_beacons.hh"
#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {

TEST(BeaconPotentialToProtoTest, pack_unpack) {
    // Setup
    const BeaconClique clique_1{.p_beacon = 0.5, .p_no_beacons = 0.1, .members = {2, 3}};
    const BeaconClique clique_2{.p_beacon = 0.6, .p_no_beacons = 0.2, .members = {4, 5}};
    const BeaconPotential pot =
        create_correlated_beacons(clique_1) * create_correlated_beacons(clique_2);

    // Action
    proto::BeaconPotential proto;
    pack_into(pot, &proto);
    const BeaconPotential unpacked = unpack_from(proto);

    // Verification
    const auto pot_members = pot.members();
    const auto unpacked_members = unpacked.members();

    // Check that they have the same members
    EXPECT_EQ(pot_members.size(), unpacked_members.size());
    for (int i = 0; i < static_cast<int>(pot_members.size()); i++) {
        EXPECT_EQ(pot_members.at(i), unpacked_members.at(i));
    }

    // Check that some probabilities are near equal
    constexpr double TOL = 1e-6;
    const std::vector<std::unordered_map<int, bool>> assignments = {
        {{2, false}, {3, false}, {4, false}, {5, false}},
        {{2, false}, {3, false}, {4, false}, {5, true}},
        {{2, false}, {3, false}, {4, true}, {5, false}},
        {{2, false}, {3, false}, {4, true}, {5, true}},
        {{2, false}, {3, true}, {4, false}, {5, false}},
        {{2, false}, {3, true}, {4, false}, {5, true}},
        {{2, false}, {3, true}, {4, true}, {5, false}},
        {{2, false}, {3, true}, {4, true}, {5, true}},
        {{2, true}, {3, false}, {4, false}, {5, false}},
        {{2, true}, {3, false}, {4, false}, {5, true}},
        {{2, true}, {3, false}, {4, true}, {5, false}},
        {{2, true}, {3, false}, {4, true}, {5, true}},
        {{2, true}, {3, true}, {4, false}, {5, false}},
        {{2, true}, {3, true}, {4, false}, {5, true}},
        {{2, true}, {3, true}, {4, true}, {5, false}},
        {{2, true}, {3, true}, {4, true}, {5, true}},
    };
    for (const auto &assignment : assignments) {
        EXPECT_NEAR(pot.log_prob(assignment), unpacked.log_prob(assignment), TOL);
    }
}
}  // namespace robot::experimental::beacon_sim
