
#include "assert/assert.hpp"

#include "gtest/gtest.h"

TEST(AssertTest, assert_test) {
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-value"
    EXPECT_DEATH(DEBUG_ASSERT(false), "Debug Assertion failed at .*");
    #pragma GCC diagnostic pop
}
