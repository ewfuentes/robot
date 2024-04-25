
#include "experimental/beacon_sim/anticorrelated_beacon_potential.hh"

#include <limits>

#include "experimental/beacon_sim/beacon_potential.hh"
#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {
TEST(AnticorrelatedBeaconPotentialTest, all_absent_is_unlikely) {
    // Setup
    BeaconPotential pot = AnticorrelatedBeaconPotential{.members = {0, 1, 2, 3, 4, 5}};

    // Action
    const double log_prob = pot.log_prob({
        {0, false},
        {1, false},
        {2, false},
        {3, false},
        {4, false},
        {5, false},
    });

    // Verification
    EXPECT_EQ(log_prob, -std::numeric_limits<double>::infinity());
}

TEST(AnticorrelatedBeaconPotentialTest, more_than_one_present_is_unlikely) {
    // Setup
    BeaconPotential pot = AnticorrelatedBeaconPotential{.members = {0, 1, 2, 3, 4, 5}};

    // Action
    const double log_prob = pot.log_prob({
        {0, false},
        {1, false},
        {2, true},
        {3, true},
        {4, false},
        {5, false},
    });

    // Verification
    EXPECT_EQ(log_prob, -std::numeric_limits<double>::infinity());
}

TEST(AnticorrelatedBeaconPotentialTest, one_present_yields_expected) {
    // Setup
    BeaconPotential pot = AnticorrelatedBeaconPotential{.members = {0, 1, 2, 3, 4, 5}};

    // Action
    const double log_prob = pot.log_prob({
        {0, false},
        {1, false},
        {2, true},
        {3, false},
        {4, false},
        {5, false},
    });

    // Verification
    EXPECT_EQ(log_prob, std::log(1.0 / pot.members().size()));
}

TEST(AnticorrelatedBeaconPotentialTest, one_present_with_partial_yields_expected) {
    // Setup
    BeaconPotential pot = AnticorrelatedBeaconPotential{.members = {0, 1, 2, 3, 4, 5}};

    // Action
    constexpr bool ALLOW_PARTIAL_ASSIGNMENT = true;
    const double log_prob = pot.log_prob(
        {
            {2, true},
        },
        ALLOW_PARTIAL_ASSIGNMENT);

    // Verification
    EXPECT_EQ(log_prob, std::log(1.0 / pot.members().size()));
}

TEST(AnticorrelatedBeaconPotentialTest, one_absent_with_partial_yields_expected) {
    // Setup
    BeaconPotential pot = AnticorrelatedBeaconPotential{.members = {0, 1, 2, 3, 4, 5}};

    // Action
    constexpr bool ALLOW_PARTIAL_ASSIGNMENT = true;
    const double log_prob = pot.log_prob(
        {
            {2, false},
        },
        ALLOW_PARTIAL_ASSIGNMENT);

    // Verification
    EXPECT_EQ(log_prob,
              std::log((pot.members().size() - 1) / static_cast<double>(pot.members().size())));
}

TEST(AnticorrelatedBeaconPotentialTest, multiple_absent_with_partial_yields_expected) {
    // Setup
    BeaconPotential pot = AnticorrelatedBeaconPotential{.members = {0, 1, 2, 3, 4, 5}};

    // Action
    constexpr bool ALLOW_PARTIAL_ASSIGNMENT = true;
    const double log_prob = pot.log_prob(
        {
            {1, false},
            {2, false},
            {3, false},
            {4, false},
            {5, false},
        },
        ALLOW_PARTIAL_ASSIGNMENT);

    // Verification
    EXPECT_EQ(log_prob, std::log(1 / static_cast<double>(pot.members().size())));
}

TEST(AnticorrelatedBeaconPotentialTest,
     one_present_with_multiple_absent_with_partial_yields_expected) {
    // Setup
    BeaconPotential pot = AnticorrelatedBeaconPotential{.members = {0, 1, 2, 3, 4, 5}};

    // Action
    constexpr bool ALLOW_PARTIAL_ASSIGNMENT = true;
    const double log_prob = pot.log_prob(
        {
            {1, true},
            {2, false},
            {5, false},
        },
        ALLOW_PARTIAL_ASSIGNMENT);

    // Verification
    EXPECT_EQ(log_prob, std::log(1 / static_cast<double>(pot.members().size())));
}
}  // namespace robot::experimental::beacon_sim
