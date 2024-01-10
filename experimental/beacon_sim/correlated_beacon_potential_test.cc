
#include "experimental/beacon_sim/correlated_beacon_potential.hh"

#include <cmath>
#include <iterator>

#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {

TEST(CorrelatedBeaconPotentialTest, compute_log_prob_test_full_assignment) {
    // Setup
    const CorrelatedBeaconPotential pot{
        .p_present = 0.25, .p_beacon_given_present = 0.1, .members = {1, 2, 3}};
    const std::vector<std::unordered_map<int, bool>> assignments = {
        {{1, false}, {2, false}, {3, false}}, {{1, false}, {2, false}, {3, true}},
        {{1, false}, {2, true}, {3, false}},  {{1, false}, {2, true}, {3, true}},
        {{1, true}, {2, false}, {3, false}},  {{1, true}, {2, false}, {3, true}},
        {{1, true}, {2, true}, {3, false}},   {{1, true}, {2, true}, {3, true}},
    };

    // Action
    constexpr double DONT_ALLOW_PARTIAL_ASSIGNMENTS = false;
    std::vector<double> probs;
    std::transform(
        assignments.begin(), assignments.end(), std::back_inserter(probs),
        [&pot](const auto &assignment) {
            return std::exp(compute_log_prob(pot, assignment, DONT_ALLOW_PARTIAL_ASSIGNMENTS));
        });

    // Verification
    const double p_000 = probs.at(0);
    const double p_001 = probs.at(1);
    const double p_010 = probs.at(2);
    const double p_011 = probs.at(3);
    const double p_100 = probs.at(4);
    const double p_101 = probs.at(5);
    const double p_110 = probs.at(6);
    const double p_111 = probs.at(7);
    constexpr double TOL = 1e-6;
    EXPECT_NEAR(p_000,
                (1 - pot.p_present) + (1 - pot.p_beacon_given_present) *
                                          (1 - pot.p_beacon_given_present) *
                                          (1 - pot.p_beacon_given_present) * pot.p_present,
                TOL);
    EXPECT_NEAR(p_111,
                pot.p_present * pot.p_beacon_given_present * pot.p_beacon_given_present *
                    pot.p_beacon_given_present,
                TOL);
    EXPECT_NEAR(p_001 + p_011 + p_101 + p_111, pot.p_beacon_given_present * pot.p_present, TOL);
    EXPECT_NEAR(p_010 + p_011 + p_110 + p_111, pot.p_beacon_given_present * pot.p_present, TOL);
    EXPECT_NEAR(p_001, p_010, TOL);
    EXPECT_NEAR(p_000 + p_001 + p_010 + p_011 + p_100 + p_101 + p_110 + p_111, 1.0, TOL);
}

TEST(CorrelatedBeaconPotentialTest, compute_log_prob_test_partial_assignment) {
    // Setup
    const CorrelatedBeaconPotential pot{
        .p_present = 0.6, .p_beacon_given_present = 0.5, .members = {1, 2, 3}};

    // Action
    std::vector<double> prob;
    const std::vector<std::unordered_map<int, bool>> assignments = {
        {{1, false}, {2, false}},
        {{1, false}, {2, true}},
        {{1, true}, {2, false}},
        {{1, true}, {2, true}},
    };
    constexpr bool ALLOW_PARTIAL_ASSIGNMENTS = true;
    for (const auto &assignment : assignments) {
        prob.push_back(std::exp(compute_log_prob(pot, assignment, ALLOW_PARTIAL_ASSIGNMENTS)));
    }

    // Verification
    const double p_00 = prob.at(0);
    const double p_01 = prob.at(1);
    const double p_10 = prob.at(2);
    const double p_11 = prob.at(3);

    constexpr double TOL = 1e-6;
    EXPECT_NEAR(p_00,
                (1 - pot.p_present) + (1 - pot.p_beacon_given_present) *
                                          (1 - pot.p_beacon_given_present) * pot.p_present,
                TOL);
    EXPECT_NEAR(p_11, pot.p_present * pot.p_beacon_given_present * pot.p_beacon_given_present, TOL);
    EXPECT_NEAR(p_01 + p_11, pot.p_beacon_given_present * pot.p_present, TOL);
    EXPECT_NEAR(p_10 + p_11, pot.p_beacon_given_present * pot.p_present, TOL);
    EXPECT_NEAR(p_01, p_10, TOL);
    EXPECT_NEAR(p_00 + p_01 + p_10 + p_11, 1.0, TOL);
}
}  // namespace robot::experimental::beacon_sim
