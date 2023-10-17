
#include "assert/assert.hpp"

#include "gtest/gtest.h"

TEST(AssertTest, assert_test) {
    EXPECT_DEATH(DEBUG_ASSERT(false), "Debug Assertion failed at .*");
}
