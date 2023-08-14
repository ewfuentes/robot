
#include "common/math/combinations.hh"

#include "gtest/gtest.h"

namespace robot::math {

TEST(NChooseKTest, n_choose_1) {
    // Action
    const auto five_choose_one = combinations(5, 1);
    const auto ten_choose_one = combinations(10, 1);

    // Verification
    EXPECT_EQ(five_choose_one.size(), 5);
    EXPECT_EQ(ten_choose_one.size(), 10);
}

TEST(NChooseKTest, n_choose_2) {
    // Action
    const auto ten_choose_two = combinations(10, 2);
    const auto five_choose_two = combinations(5, 2);

    // Verification
    EXPECT_EQ(ten_choose_two.size(), 45);
    EXPECT_EQ(five_choose_two.size(), 10);
}

TEST(NChooseKTest, n_choose_5) {
    // Action
    const auto thirty_choose_five = combinations(30, 5);
    const auto twenty_nine_choose_five = combinations(29, 5);

    // Verification
    EXPECT_EQ(thirty_choose_five.size(), 142506);
    EXPECT_EQ(twenty_nine_choose_five.size(), 118755);
}

}  // namespace robot::math
