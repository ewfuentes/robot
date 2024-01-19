
#include "experimental/beacon_sim/conditioned_potential.hh"

#include <filesystem>
#include <limits>
#include <unordered_set>

#include "Eigen/Dense"
#include "common/math/logsumexp.hh"
#include "experimental/beacon_sim/beacon_potential.hh"
#include "experimental/beacon_sim/precision_matrix_potential.hh"
#include "gtest/gtest.h"

namespace robot::experimental::beacon_sim {
TEST(ConditionedPotentialTest, get_members_test) {
    // Setup
    const BeaconPotential underlying = PrecisionMatrixPotential{
        .precision = (Eigen::Matrix2d() << 1, 0, 0, 2).finished(),
        .log_normalizer = math::logsumexp(std::vector{0.0, 1.0, 2.0, 3.0}),
        .members = {10, 20},
    };
    const auto underlying_members = underlying.members();
    const BeaconPotential conditioned = underlying.condition_on({{20, true}});

    // Action

    const auto conditioned_members = conditioned.members();

    // Verification
    EXPECT_EQ(underlying_members.size(), conditioned_members.size());

    for (const int underlying_beacon_id : underlying_members) {
        const auto iter =
            std::find(conditioned_members.begin(), conditioned_members.end(), underlying_beacon_id);
        EXPECT_NE(iter, conditioned_members.end());
    }
}

TEST(ConditionedPotentialTest, compute_log_prob_query_unconditioned_test) {
    // Setup
    const BeaconPotential underlying = PrecisionMatrixPotential{
        .precision = (Eigen::Matrix2d() << 1.0, 0.0, 0.0, 2.0).finished(),
        .log_normalizer = math::logsumexp(std::vector{0.0, 1.0, 2.0, 3.0}),
        .members = {10, 20},
    };
    const auto underlying_members = underlying.members();
    const BeaconPotential conditioned = underlying.condition_on({{20, true}});
    const BeaconPotential equivalent = PrecisionMatrixPotential{
        .precision = (Eigen::MatrixXd(1, 1) << 1.0).finished(),
        .log_normalizer = math::logsumexp(std::vector{2.0, 3.0}) - 2.0,
        .members = {10},
    };

    // Action + Verification
    const auto assignments = std::vector<std::unordered_map<int, bool>>{
        {{10, true}},
        {{10, false}},
    };
    constexpr double TOL = 1e-6;
    for (const auto &assignment : assignments) {
        EXPECT_NEAR(equivalent.log_prob(assignment), conditioned.log_prob(assignment), TOL);
    }
}

TEST(ConditionedPotentialTest, compute_log_prob_query_with_conditioned_test) {
    // Setup
    const BeaconPotential underlying = PrecisionMatrixPotential{
        .precision = (Eigen::Matrix2d() << 1.0, 0.0, 0.0, 2.0).finished(),
        .log_normalizer = math::logsumexp(std::vector{0.0, 1.0, 2.0, 3.0}),
        .members = {10, 20},
    };
    const auto underlying_members = underlying.members();
    const BeaconPotential conditioned = underlying.condition_on({{20, true}});
    const BeaconPotential equivalent = PrecisionMatrixPotential{
        .precision = (Eigen::MatrixXd(1, 1) << 1.0).finished(),
        .log_normalizer = math::logsumexp(std::vector{2.0, 3.0}) - 2.0,
        .members = {10},
    };

    // Action + Verification
    const auto assignments = std::vector<std::unordered_map<int, bool>>{
        {{20, false}, {10, true}},
        {{20, false}, {10, false}},
        {{20, true}, {10, true}},
        {{20, true}, {10, false}},
    };
    constexpr double TOL = 1e-6;
    for (const auto &assignment : assignments) {
        if (!assignment.at(20)) {
            EXPECT_EQ(conditioned.log_prob(assignment), -std::numeric_limits<double>::infinity());
        } else {
            EXPECT_NEAR(equivalent.log_prob(assignment), conditioned.log_prob(assignment), TOL);
        }
    }
}

TEST(ConditionedPotentialTest, compute_log_marginal_test) {
    // Setup
    const BeaconPotential underlying = PrecisionMatrixPotential{
        .precision = (Eigen::Matrix2d() << 1.0, 0.0, 0.0, 2.0).finished(),
        .log_normalizer = math::logsumexp(std::vector{0.0, 1.0, 2.0, 3.0}),
        .members = {10, 20},
    };
    const auto underlying_members = underlying.members();
    const BeaconPotential conditioned = underlying.condition_on({{20, true}});
    const BeaconPotential equivalent = PrecisionMatrixPotential{
        .precision = (Eigen::MatrixXd(1, 1) << 1.0).finished(),
        .log_normalizer = math::logsumexp(std::vector{2.0, 3.0}) - 2.0,
        .members = {10},
    };
    const auto present_beacon_possibilities = std::vector<std::unordered_set<int>>{
        {20, 10},
        {20},
    };

    // Action
    const auto log_marginals = conditioned.log_marginals({10});

    // Verification
    ASSERT_EQ(log_marginals.size(), present_beacon_possibilities.size());
    constexpr double TOL = 1e-6;
    for (const auto &present_beacons : present_beacon_possibilities) {
        const auto iter =
            std::find_if(log_marginals.begin(), log_marginals.end(), [&](const auto &marginal) {
                const std::unordered_set<int> marginal_set(marginal.present_beacons.begin(),
                                                           marginal.present_beacons.end());
                return marginal_set == present_beacons;
            });
        ASSERT_NE(iter, log_marginals.end());
        EXPECT_NEAR(iter->log_marginal, equivalent.log_prob(iter->present_beacons), TOL);
    }
}

TEST(ConditionedPotentialTest, compute_log_marginal_with_conditioned_test) {
    // Setup
    const BeaconPotential underlying = PrecisionMatrixPotential{
        .precision = (Eigen::Matrix2d() << 1.0, 0.0, 0.0, 2.0).finished(),
        .log_normalizer = math::logsumexp(std::vector{0.0, 1.0, 2.0, 3.0}),
        .members = {10, 20},
    };
    const auto underlying_members = underlying.members();
    const BeaconPotential conditioned = underlying.condition_on({{20, true}});
    const BeaconPotential equivalent = PrecisionMatrixPotential{
        .precision = (Eigen::MatrixXd(1, 1) << 1.0).finished(),
        .log_normalizer = math::logsumexp(std::vector{2.0, 3.0}) - 2.0,
        .members = {10},
    };
    const auto present_beacon_possibilities = std::vector<std::unordered_set<int>>{
        {20, 10},
        {20},
    };

    // Action
    const auto log_marginals = conditioned.log_marginals({10, 20});

    // Verification
    ASSERT_EQ(log_marginals.size(), present_beacon_possibilities.size());
    constexpr double TOL = 1e-6;
    for (const auto &present_beacons : present_beacon_possibilities) {
        const auto iter =
            std::find_if(log_marginals.begin(), log_marginals.end(), [&](const auto &marginal) {
                const std::unordered_set<int> marginal_set(marginal.present_beacons.begin(),
                                                           marginal.present_beacons.end());
                return marginal_set == present_beacons;
            });
        ASSERT_NE(iter, log_marginals.end());
        EXPECT_NEAR(iter->log_marginal, equivalent.log_prob(iter->present_beacons), TOL);
    }
}

TEST(ConditionedPotentialTest, compute_log_marginal_with_inconsistent_conditioned_test) {
    // Setup
    const BeaconPotential underlying = PrecisionMatrixPotential{
        .precision = (Eigen::Matrix2d() << 1.0, 0.0, 0.0, 2.0).finished(),
        .log_normalizer = math::logsumexp(std::vector{0.0, 1.0, 2.0, 3.0}),
        .members = {10, 20},
    };
    const auto underlying_members = underlying.members();
    const BeaconPotential conditioned = underlying.condition_on({{20, false}});

    // Action
    const auto log_marginals = conditioned.log_marginals({10, 20});

    // Verification
    EXPECT_TRUE(log_marginals.empty());
}

TEST(ConditionedPotentialTest, sampled_from_conditional_distribution) {
    // Setup
    const BeaconPotential underlying = PrecisionMatrixPotential{
        .precision = (Eigen::Matrix2d() << 1.0, 0.0, 0.0, 2.0).finished(),
        .log_normalizer = math::logsumexp(std::vector{0.0, 1.0, 2.0, 3.0}),
        .members = {10, 20},
    };
    const auto underlying_members = underlying.members();
    const BeaconPotential conditioned = underlying.condition_on({{20, true}});
    const BeaconPotential equivalent = PrecisionMatrixPotential{
        .precision = (Eigen::MatrixXd(1, 1) << 1.0).finished(),
        .log_normalizer = math::logsumexp(std::vector{2.0, 3.0}) - 2.0,
        .members = {10},
    };
    const auto present_beacon_possibilities = std::vector<std::vector<int>>{
        {20},
        {20, 10},
    };
    std::mt19937 gen(12345);

    // Action + Verification
    std::array<int, 2> counts = {0, 0};
    constexpr int NUM_SAMPLES = 100000;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        const auto sample = conditioned.sample(make_in_out(gen));

        {
            // Check that the condition beacon is always present
            const auto iter = std::find(sample.begin(), sample.end(), 20);
            EXPECT_NE(iter, sample.end());
        }

        {
            // Count the number of time that the other beacon is present
            const auto iter = std::find(sample.begin(), sample.end(), 10);
            const auto idx = iter == sample.end() ? 0 : 1;
            counts.at(idx)++;
        }
    }

    constexpr double TOL = 1e-2;
    for (int i = 0; i < static_cast<int>(counts.size()); i++) {
        const double approx_p = static_cast<double>(counts.at(i)) / NUM_SAMPLES;
        const double expected_p = std::exp(equivalent.log_prob(present_beacon_possibilities.at(i)));
        EXPECT_NEAR(approx_p, expected_p, TOL);
    }
}
}  // namespace robot::experimental::beacon_sim
