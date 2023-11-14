#include "common/math/n_choose_k.hh"

#include "gtest/gtest.h"

namespace robot::math {
using Param = std::tuple<int, int, int>;
class NChooseKTest : public testing::TestWithParam<Param> {};
TEST_P(NChooseKTest, n_choose_k_test) {
    // Setup
    const auto &[n, k, expected] = GetParam();

    // Action
    const int n_choose_k_result = n_choose_k(n, k);

    // Verification
    EXPECT_EQ(n_choose_k_result, expected);
}

INSTANTIATE_TEST_SUITE_P(NChooseKTestSuite, NChooseKTest,
                         testing::Values(std::make_tuple(10, 10, 1), std::make_tuple(5, 3, 10),
                                         std::make_tuple(33, 10, 92561040),
                                         std::make_tuple(33, 16, 1166803110)));

}  // namespace robot::math
