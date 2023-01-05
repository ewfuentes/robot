
#include "common/indexed_array.hh"

#include "gtest/gtest.h"

namespace robot {
namespace {
WISE_ENUM_CLASS(TestEnum, VALUE1, VALUE2, VALUE3);
WISE_ENUM_CLASS(NonContiguous, (VALUE1, 5), (VALUE2, 7), (VALUE3, 9));
}  // namespace

template <typename T>
class IndexedArrayTest : public testing::Test {};

using TestTypes = testing::Types<TestEnum, NonContiguous>;
TYPED_TEST_SUITE(IndexedArrayTest, TestTypes);

TYPED_TEST(IndexedArrayTest, list_initialization_test) {
    // Setup
    const IndexedArray<int, TypeParam> idx_arr{{
        {TypeParam::VALUE1, 10},
        {TypeParam::VALUE2, 20},
        {TypeParam::VALUE3, 30},
    }};

    // Action + Verification
    EXPECT_EQ(idx_arr[TypeParam::VALUE1], 10);
    EXPECT_EQ(idx_arr[TypeParam::VALUE2], 20);
    EXPECT_EQ(idx_arr[TypeParam::VALUE3], 30);
}

TYPED_TEST(IndexedArrayTest, value_initialization_test) {
    // Setup
    const IndexedArray<int, TypeParam> idx_arr{100};

    // Action + Verification
    EXPECT_EQ(idx_arr[TypeParam::VALUE1], 100);
    EXPECT_EQ(idx_arr[TypeParam::VALUE2], 100);
    EXPECT_EQ(idx_arr[TypeParam::VALUE3], 100);
}

TYPED_TEST(IndexedArrayTest, index_assignment_test) {
    // Setup
    IndexedArray<int, TypeParam> idx_arr;
    idx_arr[TypeParam::VALUE1] = 10;
    idx_arr[TypeParam::VALUE2] = 20;
    idx_arr[TypeParam::VALUE3] = 30;

    // Action + Verification
    EXPECT_EQ(idx_arr[TypeParam::VALUE1], 10);
    EXPECT_EQ(idx_arr[TypeParam::VALUE2], 20);
    EXPECT_EQ(idx_arr[TypeParam::VALUE3], 30);
}

TYPED_TEST(IndexedArrayTest, range_based_for_loop) {
    // Setup
    const IndexedArray<int, TypeParam> idx_arr{{
        {TypeParam::VALUE1, 10},
        {TypeParam::VALUE2, 20},
        {TypeParam::VALUE3, 30},
    }};

    // Action + Verification
    for (const auto &[idx, value] : idx_arr) {
        if (idx == TypeParam::VALUE1) {
            EXPECT_EQ(value, 10);
        } else if (idx == TypeParam::VALUE2) {
            EXPECT_EQ(value, 20);
        } else if (idx == TypeParam::VALUE3) {
            EXPECT_EQ(value, 30);
        }
    }
}

TYPED_TEST(IndexedArrayTest, range_based_for_loop_with_modify) {
    // Setup
    IndexedArray<int, TypeParam> idx_arr{{
        {TypeParam::VALUE1, 10},
        {TypeParam::VALUE2, 20},
        {TypeParam::VALUE3, 30},
    }};

    // Action
    for (auto &&[idx, value] : idx_arr) {
        value += 5;
    }

    // Verification
    for (const auto &[idx, value] : idx_arr) {
        if (idx == TypeParam::VALUE1) {
            EXPECT_EQ(value, 15);
        } else if (idx == TypeParam::VALUE2) {
            EXPECT_EQ(value, 25);
        } else if (idx == TypeParam::VALUE3) {
            EXPECT_EQ(value, 35);
        }
    }
}
}  // namespace robot
