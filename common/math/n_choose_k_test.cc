#include "common/math/n_choose_k.hh"

#include "gtest/gtest.h"

namespace robot::math {
TEST(NChooseKTest, n_choose_n_is_one) {
    // Setup
    constexpr int N = 10;

    // Action
    constexpr int n_choose_n = n_choose_k(N, N);

    // Verification
    EXPECT_EQ(n_choose_n, 1);
}

TEST(NChooseKTest, five_choose_three_is_ten) {
    // Setup
    constexpr int N = 5;
    constexpr int K = 3;

    // Action
    constexpr int five_choose_three = n_choose_k(N, K);

    // Verification
    EXPECT_EQ(five_choose_three, 10);
}
}  // namespace robot::math
