
#include "common/indexed_array.hh"

#include "gtest/gtest.h"

namespace robot {
namespace {
WISE_ENUM_CLASS(TestEnum, VALUE1, VALUE2, VALUE3);
}  // namespace

TEST(IndexedArrayTest, list_initialization_test) {
    // Setup
    const IndexedArray<int, TestEnum> idx_arr{{
        {TestEnum::VALUE1, 10},
        {TestEnum::VALUE2, 20},
        {TestEnum::VALUE3, 30},
    }};

    // Action + Verification
    EXPECT_EQ(idx_arr[TestEnum::VALUE1], 10);
    EXPECT_EQ(idx_arr[TestEnum::VALUE2], 20);
    EXPECT_EQ(idx_arr[TestEnum::VALUE3], 30);
}

TEST(IndexedArrayTest, value_initialization_test) {
    // Setup
    const IndexedArray<int, TestEnum> idx_arr{100};

    // Action + Verification
    EXPECT_EQ(idx_arr[TestEnum::VALUE1], 100);
    EXPECT_EQ(idx_arr[TestEnum::VALUE2], 100);
    EXPECT_EQ(idx_arr[TestEnum::VALUE3], 100);
}

TEST(IndexedArrayTest, index_assignment_test) {
    // Setup
    IndexedArray<int, TestEnum> idx_arr;
    idx_arr[TestEnum::VALUE1] = 10;
    idx_arr[TestEnum::VALUE2] = 20;
    idx_arr[TestEnum::VALUE3] = 30;

    // Action + Verification
    EXPECT_EQ(idx_arr[TestEnum::VALUE1], 10);
    EXPECT_EQ(idx_arr[TestEnum::VALUE2], 20);
    EXPECT_EQ(idx_arr[TestEnum::VALUE3], 30);
}

TEST(IndexedArrayTest, range_based_for_loop) {
    // Setup
    const IndexedArray<int, TestEnum> idx_arr{{
        {TestEnum::VALUE1, 10},
        {TestEnum::VALUE2, 20},
        {TestEnum::VALUE3, 30},
    }};

    // Action + Verification
    for (const auto &[idx, value] : idx_arr) {
        if (idx == TestEnum::VALUE1) {
            EXPECT_EQ(value, 10);
        } else if (idx == TestEnum::VALUE2) {
            EXPECT_EQ(value, 20);
        } else if (idx == TestEnum::VALUE3) {
            EXPECT_EQ(value, 30);
        }
    }
}

TEST(IndexedArrayTest, range_based_for_loop_with_modify) {
    // Setup
    IndexedArray<int, TestEnum> idx_arr{{
        {TestEnum::VALUE1, 10},
        {TestEnum::VALUE2, 20},
        {TestEnum::VALUE3, 30},
    }};

    // Action
    for (auto &&[idx, value] : idx_arr) {
        value += 5;
    }

    // Verification
    for (const auto &[idx, value] : idx_arr) {
        if (idx == TestEnum::VALUE1) {
            EXPECT_EQ(value, 15);
        } else if (idx == TestEnum::VALUE2) {
            EXPECT_EQ(value, 25);
        } else if (idx == TestEnum::VALUE3) {
            EXPECT_EQ(value, 35);
        }
    }
}
}  // namespace robot
