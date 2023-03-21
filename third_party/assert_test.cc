
#include "assert/assert.hpp"

#include "gtest/gtest.h"

namespace robot::third_party {
TEST(AssertTest, assert_test) {
    // Action + Verification
    VERIFY(false, "Expected true to be false");
}

}  // namespace robot::third_party
