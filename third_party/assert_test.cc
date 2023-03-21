
#include "assert/assert.hpp"

#include "gtest/gtest.h"

TEST(AssertTest, assert_test) {
    ASSERT_THROW(VERIFY(false, "Expected true to be false"), libassert::verification_failure);
}
