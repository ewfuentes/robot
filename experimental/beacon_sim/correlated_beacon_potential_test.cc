
#include "experimental/beacon_sim/correlated_beacon_potential.hh"

#include <cmath>
#include <iterator>
#include <limits>

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

TEST(CorrelatedBeaconPotentialTest, compute_conditioned_log_prob) {
    // Setup
    const CorrelatedBeaconPotential pot{.p_present = 0.6,
                                        .p_beacon_given_present = 0.5,
                                        .members = {1, 2, 3},
                                        .conditioning = {{.conditioned_members = {{1, true}}}}};

    // Action
    std::vector<double> prob;
    const std::vector<std::unordered_map<int, bool>> assignments = {
        // This is not consistent with conditioning, so it should be very low probability
        {{1, false}, {2, false}, {3, true}},
        // This implies that present is heads, so it should just be 0.25
        {{1, true}, {2, false}, {3, false}},

    };
    constexpr bool ALLOW_PARTIAL_ASSIGNMENTS = false;
    for (const auto &assignment : assignments) {
        prob.push_back(std::exp(compute_log_prob(pot, assignment, ALLOW_PARTIAL_ASSIGNMENTS)));
    }

    // Verification
    constexpr double TOL = 1e-6;
    EXPECT_EQ(prob.at(0), 0.0);
    EXPECT_NEAR(prob.at(1), pot.p_beacon_given_present * pot.p_beacon_given_present, TOL);
}

TEST(CorrelatedBeaconPotentialTest, compute_conditioned_log_marginals) {
    // Setup
    const CorrelatedBeaconPotential pot{.p_present = 0.6,
                                        .p_beacon_given_present = 0.3,
                                        .members = {1, 2, 3},
                                        .conditioning = {{.conditioned_members = {{1, true}}}}};

    // Action
    const auto log_marginals = compute_log_marginals(pot, {3});

    // Verification
    constexpr double TOL = 1e-6;
    EXPECT_EQ(log_marginals.size(), 2);
    for (const auto &marginal : log_marginals) {
        const auto iter =
            std::find(marginal.present_beacons.begin(), marginal.present_beacons.end(), 3);
        const double expected_prob = iter == marginal.present_beacons.end()
                                         ? 1 - pot.p_beacon_given_present
                                         : pot.p_beacon_given_present;
        EXPECT_NEAR(marginal.log_marginal, std::log(expected_prob), TOL);
    }
}

TEST(CorrelatedBeaconPotentialTest, compute_conditioned_log_marginals_on_conditioned) {
    // Setup
    const CorrelatedBeaconPotential pot{.p_present = 0.6,
                                        .p_beacon_given_present = 0.3,
                                        .members = {1, 2, 3},
                                        .conditioning = {{.conditioned_members = {{1, true}}}}};

    // Action
    const auto log_marginals = compute_log_marginals(pot, {1});

    // Verification
    constexpr double TOL = 1e-6;
    EXPECT_EQ(log_marginals.size(), 1);
    EXPECT_EQ(log_marginals.at(0).present_beacons.size(), 1);
    EXPECT_EQ(log_marginals.at(0).present_beacons.at(0), 1);
    EXPECT_NEAR(log_marginals.at(0).log_marginal, 0.0, TOL);
}

TEST(CorrelatedBeaconPotentialTest, generate_samples_conditioned_present) {
    // Setup
    constexpr int NUM_SAMPLES = 10000;
    const CorrelatedBeaconPotential pot{.p_present = 0.6,
                                        .p_beacon_given_present = 0.3,
                                        .members = {1, 2, 3},
                                        .conditioning = {{.conditioned_members = {{1, true}}}}};

    // Action
    std::mt19937 gen(1024);
    std::array present_count{0, 0, 0};
    for (int i = 0; i < NUM_SAMPLES; i++) {
        const auto sample = generate_sample(pot, make_in_out(gen));
        for (const int id : sample) {
            present_count[id - 1]++;
        }
    }
    // Verification
    constexpr double TOL = 1e-2;
    EXPECT_EQ(present_count.at(0), NUM_SAMPLES);
    EXPECT_NEAR(static_cast<double>(present_count.at(1)) / NUM_SAMPLES, pot.p_beacon_given_present,
                TOL);
    EXPECT_NEAR(static_cast<double>(present_count.at(2)) / NUM_SAMPLES, pot.p_beacon_given_present,
                TOL);
}

TEST(CorrelatedBeaconPotentialTest, generate_samples_conditioned_absent) {
    // Setup
    constexpr int NUM_SAMPLES = 10000;
    const CorrelatedBeaconPotential pot{.p_present = 0.6,
                                        .p_beacon_given_present = 0.3,
                                        .members = {1, 2, 3},
                                        .conditioning = {{.conditioned_members = {{1, false}}}}};

    // Action
    std::mt19937 gen(1024);
    std::array present_count{0, 0, 0};
    for (int i = 0; i < NUM_SAMPLES; i++) {
        const auto sample = generate_sample(pot, make_in_out(gen));
        for (const int id : sample) {
            present_count[id - 1]++;
        }
    }
    // Verification
    constexpr double TOL = 1e-2;
    EXPECT_EQ(present_count.at(0), 0);
    EXPECT_NEAR(static_cast<double>(present_count.at(1)) / NUM_SAMPLES,
                pot.p_beacon_given_present * pot.p_present, TOL);
    EXPECT_NEAR(static_cast<double>(present_count.at(2)) / NUM_SAMPLES,
                pot.p_beacon_given_present * pot.p_present, TOL);
}

TEST(CorrelatedBeaconPotentialTest, generate_samples) {
    // Setup
    constexpr int NUM_SAMPLES = 10000;
    const CorrelatedBeaconPotential pot{
        .p_present = 0.6, .p_beacon_given_present = 0.3, .members = {1, 2, 3}, .conditioning = {}};

    // Action
    std::mt19937 gen(1024);
    std::array present_count{0, 0, 0};
    for (int i = 0; i < NUM_SAMPLES; i++) {
        const auto sample = generate_sample(pot, make_in_out(gen));
        for (const int id : sample) {
            present_count[id - 1]++;
        }
    }
    // Verification
    constexpr double TOL = 1e-2;
    EXPECT_NEAR(static_cast<double>(present_count.at(0)) / NUM_SAMPLES,
                pot.p_beacon_given_present * pot.p_present, TOL);
    EXPECT_NEAR(static_cast<double>(present_count.at(1)) / NUM_SAMPLES,
                pot.p_beacon_given_present * pot.p_present, TOL);
    EXPECT_NEAR(static_cast<double>(present_count.at(2)) / NUM_SAMPLES,
                pot.p_beacon_given_present * pot.p_present, TOL);
}
}  // namespace robot::experimental::beacon_sim
