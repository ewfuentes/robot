
#include "common/math/sample_without_replacement.hh"

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"

namespace robot::math {

TEST(SampleWithoutReplacement, samples_match_dist) {
    // Setup
    std::vector<double> weights = {1, 2, 3, 4, 5};
    constexpr int NUM_SAMPLES = 2;
    constexpr int NUM_EXPERIMENTS = 100000;
    std::mt19937 gen(0);

    // Action
    absl::flat_hash_map<int, int> counts_by_digit;
    for (int i = 0; i < NUM_EXPERIMENTS; i++) {
        constexpr bool NOT_LOG_WEIGHTS = false;
        const auto sample = reservoir_sample_without_replacement(weights, NUM_SAMPLES, NOT_LOG_WEIGHTS, make_in_out(gen));
        for (const int elem: sample) {
            if (counts_by_digit.find(elem) == counts_by_digit.end()) {
              counts_by_digit[elem] = 0;
            }
            counts_by_digit[elem] += 1;
        }
    }
    
    // Verification
    // Ideally we would compare the empirical distribution to the expected distribution
    // but computing the probability with non-uniform weights is escaping me at the moment.
    for (int i = 0; i < static_cast<int>(weights.size()); i++) {
        for (int j = i+1; j < static_cast<int>(weights.size()); j++) {
            EXPECT_LT(counts_by_digit.at(i), counts_by_digit.at(j));
        }
    }
}

TEST(SampleWithoutReplacement, samples_match_dist_log_weights) {
    // Setup
    std::vector<double> weights = {-2, -1, 0, 1, 2};
    constexpr int NUM_SAMPLES = 2;
    constexpr int NUM_EXPERIMENTS = 100000;
    std::mt19937 gen(0);

    // Action
    absl::flat_hash_map<int, int> counts_by_digit;
    for (int i = 0; i < NUM_EXPERIMENTS; i++) {
        constexpr bool LOG_WEIGHTS = true;
        const auto sample = reservoir_sample_without_replacement(weights, NUM_SAMPLES, LOG_WEIGHTS, make_in_out(gen));
        for (const int elem: sample) {
            if (counts_by_digit.find(elem) == counts_by_digit.end()) {
              counts_by_digit[elem] = 0;
            }
            counts_by_digit[elem] += 1;
        }
    }
    
    // Verification
    // Ideally we would compare the empirical distribution to the expected distribution
    // but computing the probability with non-uniform weights is escaping me at the moment.
    for (int i = 0; i < static_cast<int>(weights.size()); i++) {
        for (int j = i+1; j < static_cast<int>(weights.size()); j++) {
            EXPECT_LT(counts_by_digit.at(i), counts_by_digit.at(j));
        }
    }
}
}
